import numpy as np
import open3d as o3d
from cpdalp import DeformableRegistration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time
import gc
import os
import json
import trimesh


# This is the ALPACA class that performs the alignment of the source landmarks to the target mesh. 
# This version was written by Alejandro Gutierrez and was based on the original ALPACA code by Arthur Porto. 
# The original code can be found at https://github.com/SlicerMorph/SlicerMorph/blob/master/ALPACA/ALPACA.py
# Portions Copyright (c) 2019, SlicerMorph Project. All rights reserved.
# Those portions are used under the BSD 2-Clause License; see
# third_party_licenses/SlicerMorph-BSD-2-Clause.txt for the full notice.

# Fixed Poisson-disk budget for alignment: reference always uses this many samples; the
# subject uses the same count scaled by incoming vertex ratio (target_vertices/source_vertices).
ALIGNMENT_REFERENCE_POISSON_POINTS = 5000


class ALPACA:
    def __init__(self):
        # Default parameters based on ALPACA
        self.params = {
            "point_density": 0.5,
            "normal_search_radius": 1,
            "fpfh_search_radius": 2.5,
            "distance_threshold": 0.5,
            # Tuned for RMS-normalized clouds (~unit scale). Old 0.03–0.07 / 0.1 were for Frobenius norm.
            "max_ransac_iter": 2000000,
            "ransac_confidence": 0.9999,
            "alpha": 2.0,  # CPD rigidity parameter
            "beta": 2.0,   # CPD motion coherence
            "cpd_iterations": 100,
            "cpd_tolerance": 0.0001
        }

    def create_landmarks_from_mesh(self, vertices, faces, n_landmarks=0):
        """
        Creates evenly distributed landmarks on a mesh surface
        
        Args:
            vertices: np.array of shape (N, 3) containing vertex coordinates
            faces: np.array of shape (M, 3) containing face indices
            n_landmarks: Number of landmarks to generate
            
        Returns:
            np.array of shape (n_landmarks, 3) containing landmark coordinates
        """
        # Convert to Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Compute mesh normals
        mesh.compute_vertex_normals()

        # Default count: fixed budget (same as align_landmarks_to_mesh reference path).
        if n_landmarks == 0:
            n_landmarks = ALIGNMENT_REFERENCE_POISSON_POINTS
        print("n_landmarks: "+str(n_landmarks))
        
        # Sample points uniformly
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_landmarks)
        
        return np.asarray(pcd.points)

    def calculate_landmark_distances(self, landmarks, vertices, faces):
        """
        Calculate the shortest distance between each landmark and the mesh surface
        using Open3D's optimized distance computation.

        Args:
            landmarks: (N, 3) array of landmark coordinates
            vertices: (M, 3) array of mesh vertex coordinates
            faces: (K, 3) array of face indices

        Returns:
            numpy array of float - shortest distances from each landmark to the mesh surface
        """
        # Create an Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Create a Scene with the mesh for distance computation
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)

        # Convert landmarks to tensor
        points = o3d.core.Tensor(landmarks.astype(np.float32))
        
        # Compute closest points and distances
        result = scene.compute_closest_points(points)
        closest_points = result['points'].numpy()
        
        # Calculate distances
        distances = np.linalg.norm(landmarks - closest_points, axis=1)

        return distances

    @staticmethod
    def apply_rigid_transform(points, matrix_4x4):
        """Apply a 4x4 rigid transform to Nx3 points."""
        points = np.asarray(points, dtype=np.float64)
        homogeneous = np.c_[points, np.ones((len(points), 1))]
        return (matrix_4x4 @ homogeneous.T).T[:, :3]

    @staticmethod
    def compose_source_to_target_transform(
        chosen_transform, source_centroid, target_centroid, normalization_scale_factor
    ):
        """
        Build a 4x4 rigid transform mapping original source points into original target space.

        The alignment pipeline centers both clouds, RMS-normalizes by normalization_scale_factor,
        then applies chosen_transform (source-normalized -> target-normalized).
        """
        R = chosen_transform[:3, :3]
        t = chosen_transform[:3, 3]
        source_centroid = np.asarray(source_centroid, dtype=np.float64)
        target_centroid = np.asarray(target_centroid, dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = (
            normalization_scale_factor * t
            + target_centroid
            - R @ source_centroid
        )
        return T

    def calculate_landmarks_closest_points(self, landmarks, vertices, faces):
        # Create an Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Create a Scene with the mesh for distance computation
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)

        # Convert landmarks to tensor
        points = o3d.core.Tensor(landmarks.astype(np.float32))

        # Compute closest points and distances
        result = scene.compute_closest_points(points)
        closest_points = result['points'].numpy()

        return closest_points

    def _weld_coincident_vertices(self, vertices, faces, relative_eps=1e-9):
        """
        Merge vertices that share the same position so triangle adjacency is
        index-based (Open3D cluster_connected_triangles / hole healing).

        Exported PLYs often store one unique vertex triple per face (V == 3F).
        Without welding, every triangle is its own component and the small-
        component prune can delete the entire outer surface.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            return vertices, faces

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        extent = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))
        eps = max(relative_eps * max(extent, 1e-12), 1e-12)
        n_before = len(mesh.vertices)
        mesh.merge_close_vertices(eps)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        n_after = len(mesh.vertices)
        if n_after < n_before:
            print(
                f"Welded coincident vertices: {n_before} -> {n_after} "
                f"(eps={eps:.3e})"
            )
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    def get_outer_mesh(self, vertices, faces, invert_normals=False):
        # -------------------------------------------------------------------
        # Select the outer-surface extraction method:
        #   1 = Ray-cast per-triangle (original)
        #   2 = Ray-cast + small-component cleanup
        #   3 = Centroid-outward normal voting (no ray-cast)
        #   4 = Multi-view visibility sampling (most precise, slower)
        # -------------------------------------------------------------------
        # Preserved PLYs may be unwelded (one vertex triple per face). Weld first
        # so connectivity-based cleanup and hole healing see a real surface.
        if not invert_normals:
            vertices, faces = self._weld_coincident_vertices(vertices, faces)

        outer_mesh_method = 4

        if outer_mesh_method == 1:
            # ----------------------------------------------------------------
            # Method 1 (original): single outward ray per triangle; keep
            # triangles whose ray escapes without hitting another face.
            # ----------------------------------------------------------------
            outer_mesh = o3d.geometry.TriangleMesh()
            outer_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if invert_normals:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1])
            else:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces)
            outer_mesh.compute_triangle_normals()

            normals = np.asarray(outer_mesh.triangle_normals)
            triangles = np.asarray(outer_mesh.triangles)
            verts = np.asarray(outer_mesh.vertices)

            centers = np.mean(verts[triangles], axis=1)
            del verts, triangles

            epsilon = 1e-3
            ray_origins = centers + epsilon * normals
            ray_directions = normals
            del centers

            rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)
            del ray_origins, ray_directions

            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(outer_mesh)
            scene.add_triangles(mesh_t)
            del mesh_t

            result = scene.cast_rays(rays)
            hit_distances = result['t_hit'].numpy()
            del rays, scene, result

            outer_indices = np.where(np.isinf(hit_distances))[0]
            outer_faces = faces[outer_indices]
            del hit_distances, outer_indices

            outer_mesh.triangles = o3d.utility.Vector3iVector(outer_faces)
            del outer_faces, normals

            outer_mesh = outer_mesh.remove_unreferenced_vertices()

            if len(outer_mesh.vertices) < 0.02 * len(vertices) and not invert_normals:
                print("Reduction of vertices is more than 98% (" + str(len(outer_mesh.vertices) / len(vertices)) + "), trying again with inverted normals")
                del outer_mesh
                gc.collect()
                time.sleep(1)
                return self.get_outer_mesh(vertices, faces, invert_normals=True)

            return np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles)

        elif outer_mesh_method == 2:
            # ----------------------------------------------------------------
            # Method 2: Ray-cast (same as method 1) then keep only the
            # small disconnected components to remove inner-surface speckles.
            # ----------------------------------------------------------------
            outer_mesh = o3d.geometry.TriangleMesh()
            outer_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if invert_normals:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1])
            else:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces)
            outer_mesh.compute_triangle_normals()

            normals = np.asarray(outer_mesh.triangle_normals)
            triangles = np.asarray(outer_mesh.triangles)
            verts = np.asarray(outer_mesh.vertices)

            centers = np.mean(verts[triangles], axis=1)
            del verts, triangles

            epsilon = 1e-3
            ray_origins = centers + epsilon * normals
            ray_directions = normals
            del centers

            rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)
            del ray_origins, ray_directions

            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(outer_mesh)
            scene.add_triangles(mesh_t)
            del mesh_t

            result = scene.cast_rays(rays)
            hit_distances = result['t_hit'].numpy()
            del rays, scene, result

            outer_indices = np.where(np.isinf(hit_distances))[0]
            outer_faces = faces[outer_indices]
            del hit_distances, outer_indices

            outer_mesh.triangles = o3d.utility.Vector3iVector(outer_faces)
            del outer_faces, normals

            outer_mesh = outer_mesh.remove_unreferenced_vertices()

            if len(outer_mesh.vertices) < 0.02 * len(vertices) and not invert_normals:
                print("Reduction of vertices is more than 98% (" + str(len(outer_mesh.vertices) / len(vertices)) + "), trying again with inverted normals")
                del outer_mesh
                gc.collect()
                time.sleep(1)
                return self.get_outer_mesh(vertices, faces, invert_normals=True)

            outer_mesh = self._prune_small_connected_components(
                outer_mesh, method_label="Method 2"
            )

            return np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles)

        elif outer_mesh_method == 3:
            # ----------------------------------------------------------------
            # Method 3: Centroid-outward normal voting — no ray-casting.
            # Each triangle whose normal points away from the mesh centroid
            # (positive dot product) is classified as outer. Works well for
            # convex-ish shapes (skulls, long bones) where the global centroid
            # is reliably inside the structure.
            # ----------------------------------------------------------------
            outer_mesh = o3d.geometry.TriangleMesh()
            outer_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if invert_normals:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1])
            else:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces)
            outer_mesh.compute_triangle_normals()

            normals = np.asarray(outer_mesh.triangle_normals)
            triangles = np.asarray(outer_mesh.triangles)
            verts = np.asarray(outer_mesh.vertices)

            centers = np.mean(verts[triangles], axis=1)
            centroid = verts.mean(axis=0)
            del verts, triangles

            # Vector from mesh centroid to each triangle centre
            outward_vectors = centers - centroid
            del centers

            # Positive dot product → normal points away from centroid → outer face
            dot_products = np.einsum('ij,ij->i', normals, outward_vectors)
            del outward_vectors, normals

            outer_indices = np.where(dot_products > 0)[0]
            outer_faces = faces[outer_indices]
            del dot_products, outer_indices

            outer_mesh.triangles = o3d.utility.Vector3iVector(outer_faces)
            del outer_faces

            outer_mesh = outer_mesh.remove_unreferenced_vertices()

            if len(outer_mesh.vertices) < 0.02 * len(vertices) and not invert_normals:
                print("Reduction of vertices is more than 98% (" + str(len(outer_mesh.vertices) / len(vertices)) + "), trying again with inverted normals")
                del outer_mesh
                gc.collect()
                time.sleep(1)
                return self.get_outer_mesh(vertices, faces, invert_normals=True)

            outer_mesh = self._prune_small_connected_components(
                outer_mesh, method_label="Method 3"
            )

            return np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles)

        elif outer_mesh_method == 4:
            # ----------------------------------------------------------------
            # Method 4: Multi-view visibility sampling.
            # Sample viewpoints on a sphere surrounding the mesh, cast rays
            # from each viewpoint to every triangle centre, and keep only
            # triangles that are the first hit from at least one viewpoint.
            # Inner surfaces (fully occluded from every direction) get zero
            # votes and are discarded. Concavities remain because they are
            # visible from some angles. Based on the Hidden Point Removal
            # (HPR) principle (Katz, Tal & Basri, 2007).
            # ----------------------------------------------------------------
            n_viewpoints = 64

            outer_mesh = o3d.geometry.TriangleMesh()
            outer_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if invert_normals:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1])
            else:
                outer_mesh.triangles = o3d.utility.Vector3iVector(faces)

            verts = np.asarray(outer_mesh.vertices)
            tris = np.asarray(outer_mesh.triangles)
            n_triangles = len(tris)

            # Triangle centres — the targets we cast rays toward
            tri_centers = np.mean(verts[tris], axis=1).astype(np.float32)

            # Unnormalized per-triangle normals (direction only matters for
            # backface test). Winding matches `tris`, consistent with
            # `invert_normals` retry below.
            v0 = verts[tris[:, 0]]
            tri_normals = np.cross(
                verts[tris[:, 1]] - v0,
                verts[tris[:, 2]] - v0,
            ).astype(np.float32)
            del v0

            # Build the ray-casting scene once
            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(outer_mesh)
            scene.add_triangles(mesh_t)
            del mesh_t

            # Bounding sphere: centre + radius with padding
            mesh_center = verts.mean(axis=0)
            mesh_radius = float(np.linalg.norm(verts - mesh_center, axis=1).max())
            sphere_radius = mesh_radius * 1.5

            # Fibonacci sphere for even viewpoint distribution
            viewpoints = np.empty((n_viewpoints, 3), dtype=np.float64)
            golden_ratio = (1 + np.sqrt(5)) / 2
            for idx in range(n_viewpoints):
                theta = np.arccos(1 - 2 * (idx + 0.5) / n_viewpoints)
                phi = 2 * np.pi * idx / golden_ratio
                viewpoints[idx] = mesh_center + sphere_radius * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])

            visibility_count = np.zeros(n_triangles, dtype=np.int32)

            # Rays parametrised so the target centre is at t = 1 (direction =
            # tri_center - vp, unnormalized). test_occlusions uses any-hit
            # (faster than cast_rays nearest-hit). Preallocate once.
            rays_buf = np.empty((n_triangles, 6), dtype=np.float32)
            eps = 1e-4

            for vp in viewpoints:
                vp_f32 = vp.astype(np.float32)

                # Backface culling: a backfacing triangle cannot be the first
                # hit along a ray from vp, so it cannot receive a vote.
                to_vp = vp_f32 - tri_centers
                frontfacing = np.einsum('ij,ij->i', tri_normals, to_vp) > 0
                idx_front = np.flatnonzero(frontfacing)
                k = idx_front.size
                if k == 0:
                    continue

                rays_buf[:k, 0:3] = vp_f32
                np.subtract(tri_centers[idx_front], vp_f32, out=rays_buf[:k, 3:6])

                rays_t = o3d.core.Tensor(rays_buf[:k])
                occluded = scene.test_occlusions(
                    rays_t, tnear=0.0, tfar=1.0 - eps
                ).numpy()
                visible_local = ~occluded

                np.add.at(visibility_count, idx_front[visible_local], 1)

            del scene, tri_centers, tri_normals, rays_buf, verts, tris

            # --- Intelligent cutoff: bimodal valley or uniform fallback ---
            # For open/complex objects, visibility is often bimodal and the
            # first-valley heuristic works well. For near-uniform "blob-like"
            # meshes, the histogram can be unimodal and valley thresholds may
            # over-prune catastrophically, so we use a conservative percentile.
            max_count = visibility_count.max()
            nonzero_count = np.count_nonzero(visibility_count)
            zero_count = n_triangles - nonzero_count
            print(f"Method 4: visibility range [{visibility_count.min()}, {max_count}], "
                  f"mean={visibility_count.mean():.1f}, median={np.median(visibility_count):.1f}, "
                  f"zero={zero_count}, nonzero={nonzero_count}")

            if max_count > 0:
                raw_hist = np.bincount(visibility_count, minlength=max_count + 1).astype(np.float64)

                # Smooth the histogram to find stable peaks/valleys
                kernel_size = max(3, (max_count + 1) // 10) | 1  # odd
                kernel = np.ones(kernel_size) / kernel_size
                smoothed = np.convolve(raw_hist, kernel, mode='same')

                # 1) Find the first significant peak in the lower half
                search_end = max(2, max_count // 2)
                first_peak_idx = np.argmax(smoothed[1:search_end]) + 1  # skip bin 0

                # 2) From that peak, walk right until the smoothed histogram
                #    starts rising again — that's the valley
                valley_idx = first_peak_idx
                for i in range(first_peak_idx + 1, len(smoothed)):
                    if smoothed[i] < smoothed[valley_idx]:
                        valley_idx = i
                    elif smoothed[i] > smoothed[valley_idx] * 1.3:
                        break

                peak_height = smoothed[first_peak_idx] if first_peak_idx < len(smoothed) else 0.0
                valley_height = smoothed[valley_idx] if valley_idx < len(smoothed) else peak_height
                valley_depth = (peak_height - valley_height) / max(peak_height, 1e-8)
                right_mass = smoothed[valley_idx + 1:] if valley_idx + 1 < len(smoothed) else np.array([])
                right_rebound = right_mass.size > 0 and np.max(right_mass) > valley_height * 1.3
                has_bimodal_signal = (
                    valley_idx > first_peak_idx + 1
                    and valley_depth >= 0.2
                    and right_rebound
                )

                nonzero_votes = visibility_count[visibility_count > 0]
                percentile_threshold = int(np.ceil(np.percentile(nonzero_votes, 2))) if len(nonzero_votes) > 0 else 1
                percentile_threshold = max(1, min(percentile_threshold, max_count))

                if has_bimodal_signal:
                    min_votes = max(1, valley_idx)
                    threshold_mode = "bimodal-valley"
                    print(f"Method 4: first peak at bin {first_peak_idx}, valley at bin {valley_idx}, "
                          f"depth={valley_depth:.2f}, threshold = {min_votes} (of {n_viewpoints} viewpoints)")
                else:
                    min_votes = percentile_threshold
                    threshold_mode = "uniform-percentile"
                    print(f"Method 4: weak bimodality (peak={first_peak_idx}, valley={valley_idx}, "
                          f"depth={valley_depth:.2f}); using p2 threshold = {min_votes} "
                          f"(of {n_viewpoints} viewpoints)")
            else:
                min_votes = 1
                threshold_mode = "degenerate"

            outer_indices = np.where(visibility_count >= min_votes)[0]
            kept_ratio = len(outer_indices) / max(1, n_triangles)
            if kept_ratio < 0.01 and max_count > 0:
                nonzero_votes = visibility_count[visibility_count > 0]
                safe_threshold = int(np.ceil(np.percentile(nonzero_votes, 2))) if len(nonzero_votes) > 0 else 1
                safe_threshold = max(1, min(safe_threshold, max_count))
                if safe_threshold < min_votes:
                    print(f"Method 4: retained ratio {kept_ratio:.5f} too low with {threshold_mode} threshold "
                          f"({min_votes}); falling back to safe p2 threshold {safe_threshold}")
                    min_votes = safe_threshold
                    threshold_mode = "safety-percentile"
                    outer_indices = np.where(visibility_count >= min_votes)[0]
                    kept_ratio = len(outer_indices) / max(1, n_triangles)

            print(f"Method 4: {len(outer_indices)} / {n_triangles} triangles kept (>= {min_votes} votes, "
                  f"mode={threshold_mode}, ratio={kept_ratio:.4f})")

            del visibility_count

            outer_mesh.triangles = o3d.utility.Vector3iVector(faces[outer_indices])
            del outer_indices

            outer_mesh = outer_mesh.remove_unreferenced_vertices()

            if len(outer_mesh.vertices) < 0.02 * len(vertices) and not invert_normals:
                print("Reduction of vertices is more than 98% (" + str(len(outer_mesh.vertices) / len(vertices)) + "), trying again with inverted normals")
                del outer_mesh
                gc.collect()
                time.sleep(1)
                return self.get_outer_mesh(vertices, faces, invert_normals=True)

            # Drop only tiny disconnected islands before hole healing. Medium/large
            # isolated regions are kept; running this after healing can fuse
            # inner-surface fragments to the outer shell via bridged triangles.
            outer_mesh = self._prune_small_connected_components(
                outer_mesh, method_label="Method 4"
            )

            out_v = np.asarray(outer_mesh.vertices)
            out_f = np.asarray(outer_mesh.triangles)
            out_v, out_f = self._heal_small_holes(out_v, out_f, vertices, faces)
            return out_v, out_f

        else:
            raise ValueError(f"Unknown outer_mesh_method: {outer_mesh_method}. Choose 1, 2, 3, or 4.")

        

    def _prune_small_connected_components(
        self, mesh, min_component_fraction=0.005, method_label=""
    ):
        """
        Remove tiny disconnected triangle islands after outer-surface extraction.

        Components whose triangle count is below ``min_component_fraction`` of
        the total are discarded; all medium/large regions are kept even when
        they are not connected to the largest component.
        """
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) <= 1:
            return mesh

        total_triangles = int(cluster_n_triangles.sum())
        min_triangles = max(1, int(np.ceil(min_component_fraction * total_triangles)))
        keep_clusters = cluster_n_triangles >= min_triangles

        if keep_clusters.all():
            print(
                f"{method_label}: all {len(cluster_n_triangles)} components above "
                f"min size ({min_triangles} triangles, "
                f"{min_component_fraction:.3%} of total)"
            )
            return mesh

        kept_triangles = int(cluster_n_triangles[keep_clusters].sum())
        # Unwelded meshes report one component per triangle; never empty the mesh.
        if kept_triangles == 0:
            print(
                f"{method_label}: component prune would remove all "
                f"{total_triangles} triangles (likely disconnected/unwelded faces); "
                f"keeping unpruned mesh"
            )
            return mesh

        removed_components = int((~keep_clusters).sum())
        keep_cluster_ids = np.where(keep_clusters)[0]
        triangles_to_keep = np.isin(triangle_clusters, keep_cluster_ids)
        mesh.remove_triangles_by_mask(~triangles_to_keep)
        mesh = mesh.remove_unreferenced_vertices()
        print(
            f"{method_label}: removed {removed_components} small component(s), "
            f"kept {int(keep_clusters.sum())}/{len(cluster_n_triangles)} components "
            f"({kept_triangles}/{total_triangles} triangles, "
            f"min={min_component_fraction:.3%})"
        )
        return mesh

    def _heal_small_holes(self, outer_verts, outer_faces,
                          original_verts, original_faces,
                          max_perimeter_fraction=0.8,
                          partial_heal_iterations=4, # 4 default
                          partial_heal_extent=2.0,
                          max_iterations=100): # 100 default
        """
        Fast hole healing: recover original-mesh triangles from boundary
        edges inward one ring at a time, using precomputed adjacency for
        local face queries instead of full-mesh scans.

        Parameters
        ----------
        outer_verts, outer_faces : np.ndarray
            Output of Method 4.
        original_verts, original_faces : np.ndarray
            Full (inner+outer) mesh passed to get_outer_mesh.
        max_perimeter_fraction : float
            Loops with perimeter at or below this fraction of the bounding-box
            diagonal are healed until closed (full heal).
        partial_heal_iterations : int
            For loops between ``max_perimeter`` and
            ``max_perimeter * partial_heal_extent``, only the first this many
            waves add one triangle ring each (partial inward fill, e.g. mouth rim).
            Set to 0 to disable partial healing.
        partial_heal_extent : float
            Loops with perimeter above
            ``max_perimeter_fraction * partial_heal_extent * bbox_diag`` are
            never healed (e.g. crop plane). Must be > 1.0 to leave a gap
            between full-heal and never-heal bands.
        max_iterations : int
            Maximum outer recovery waves (reclassifies boundary loops each wave).
        """
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        outer_faces = outer_faces.astype(np.int32)
        original_verts = original_verts.astype(np.float32)
        original_faces = original_faces.astype(np.int32)

        # ── 1. Map outer vertices → original vertex indices ───────────────
        tree = cKDTree(original_verts)
        _, outer_to_orig = tree.query(outer_verts)
        outer_to_orig = outer_to_orig.astype(np.int32)

        n_orig_verts = len(original_verts)
        n_orig_faces = len(original_faces)

        # ── 2. Build active face mask (in terms of original face indices) ──
        mapped = outer_to_orig[outer_faces]
        mapped_sorted = np.sort(mapped, axis=1)
        orig_sorted = np.sort(original_faces, axis=1)

        def _hash_faces(f):
            return ((f[:, 0].astype(np.int64) * n_orig_verts +
                    f[:, 1].astype(np.int64)) * n_orig_verts +
                    f[:, 2].astype(np.int64))

        outer_hashes = _hash_faces(mapped_sorted)
        orig_hashes = _hash_faces(orig_sorted)

        order = np.argsort(orig_hashes)
        sorted_hashes = orig_hashes[order]
        pos = np.searchsorted(sorted_hashes, outer_hashes)
        pos = np.clip(pos, 0, len(sorted_hashes) - 1)
        match = sorted_hashes[pos] == outer_hashes
        active_mask = np.zeros(n_orig_faces, dtype=bool)
        active_mask[order[pos[match]]] = True

        n_initial = int(active_mask.sum())

        # ── 3. Precompute mesh adjacency (once) ───────────────────────────
        f = original_faces
        e0 = np.sort(f[:, [0, 1]], axis=1)
        e1 = np.sort(f[:, [1, 2]], axis=1)
        e2 = np.sort(f[:, [0, 2]], axis=1)
        all_edge_verts = np.vstack([e0, e1, e2])
        edge_keys = (all_edge_verts[:, 0].astype(np.int64) * n_orig_verts +
                     all_edge_verts[:, 1])
        unique_edge_keys, edge_inverse = np.unique(
            edge_keys, return_inverse=True
        )
        n_edges = len(unique_edge_keys)
        face_edge_ids = edge_inverse.reshape(3, n_orig_faces).T

        edge_v0 = (unique_edge_keys // n_orig_verts).astype(np.int32)
        edge_v1 = (unique_edge_keys % n_orig_verts).astype(np.int32)

        face_for_edge = np.repeat(np.arange(n_orig_faces, dtype=np.int32), 3)
        vert_face_mat = csr_matrix(
            (np.ones(3 * n_orig_faces, dtype=np.int8),
             (f.ravel(), face_for_edge)),
            shape=(n_orig_verts, n_orig_faces),
        )

        edge_active_count = np.zeros(n_edges, dtype=np.int8)
        active_face_ids = np.flatnonzero(active_mask)
        if active_face_ids.size:
            np.add.at(
                edge_active_count,
                face_edge_ids[active_face_ids].ravel(),
                1,
            )

        def _activate_faces(face_ids):
            if face_ids.size == 0:
                return 0
            active_mask[face_ids] = True
            np.add.at(edge_active_count, face_edge_ids[face_ids].ravel(), 1)
            return int(face_ids.size)

        def _classify_boundary_loops():
            """Return healable boundary-vertex mask, or None if no boundary."""
            bnd_edge_ids = np.flatnonzero(edge_active_count == 1)
            if bnd_edge_ids.size == 0:
                return None

            be = np.column_stack([edge_v0[bnd_edge_ids],
                                  edge_v1[bnd_edge_ids]])
            boundary_vert_ids = np.unique(be.ravel())
            n_bv = boundary_vert_ids.size

            r = np.searchsorted(boundary_vert_ids, be[:, 0])
            c = np.searchsorted(boundary_vert_ids, be[:, 1])
            adj_mat = csr_matrix(
                (np.ones(len(r), dtype=np.int8), (r, c)),
                shape=(n_bv, n_bv),
            )
            adj_mat = adj_mat + adj_mat.T

            n_components, labels = connected_components(
                adj_mat, directed=False
            )

            edge_lengths = np.linalg.norm(
                original_verts[be[:, 0]] - original_verts[be[:, 1]],
                axis=1,
            )
            comp_perimeters = np.zeros(n_components, dtype=np.float64)
            np.add.at(comp_perimeters, labels[r], edge_lengths)

            if use_perimeter_filter:
                small_components = np.where(comp_perimeters <= max_perimeter)[0]
                partial_components = np.where(
                    (comp_perimeters > max_perimeter)
                    & (comp_perimeters <= giant_perimeter)
                )[0]
                if (
                    partial_heal_iterations > 0
                    and iteration < partial_heal_iterations
                ):
                    active_components = np.unique(
                        np.concatenate([small_components, partial_components])
                    )
                else:
                    active_components = small_components
            else:
                active_components = np.arange(n_components)

            if active_components.size == 0:
                return None

            healable_mask = np.zeros(n_orig_verts, dtype=bool)
            healable_mask[boundary_vert_ids[np.isin(labels, active_components)]] = True
            return healable_mask

        # ── 4. Perimeter thresholds ───────────────────────────────────────
        bbox_diag = np.linalg.norm(
            original_verts.max(axis=0) - original_verts.min(axis=0)
        )
        max_perimeter = max_perimeter_fraction * bbox_diag
        giant_perimeter = max_perimeter * partial_heal_extent

        use_perimeter_filter = (
            max_perimeter_fraction < 1.0 or partial_heal_iterations > 0
        )

        # ── 5. One ring per wave (original semantics), local face queries ───
        for iteration in range(max_iterations):
            healable_mask = _classify_boundary_loops()
            if healable_mask is None or not healable_mask.any():
                break

            healable_bv_ids = np.flatnonzero(healable_mask)
            sub = vert_face_mat[healable_bv_ids]
            face_bv_count = np.asarray(sub.sum(axis=0)).ravel()
            candidates = np.flatnonzero(
                (~active_mask) & (face_bv_count >= 2)
            )
            if candidates.size == 0:
                break

            wave_added = _activate_faces(candidates)
            print(f"  Wave {iteration + 1}: +{wave_added} triangles "
                  f"({healable_bv_ids.size} boundary vertices)")

        n_recovered = int(active_mask.sum()) - n_initial
        print(f"Hole healing: recovered {n_recovered} triangles "
              f"({n_initial} → {int(active_mask.sum())})")

        # ── 6. Rebuild mesh ───────────────────────────────────────────────
        final_faces = original_faces[active_mask]
        used_verts = np.unique(final_faces.ravel())
        remap = np.full(n_orig_verts, -1, dtype=np.int32)
        remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)

        new_verts = original_verts[used_verts]
        new_faces = remap[final_faces]

        return new_verts, new_faces

    def plot_debug_step(self, source_points, target_points, title, output_dir=None):
        """
        Helper function to plot point clouds at each step and save to scan directory
        
        Args:
            source_points: np.array of source landmarks
            target_points: np.array of target landmarks  
            title: string title for the plot
            output_dir: directory to save the image (if None, saves to current dir)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create output directory if specified and doesn't exist
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'debug_{title.replace(" ", "_")}.jpg')
            else:
                output_path = f'debug_{title.replace(" ", "_")}.jpg'
            
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot source points in red
            ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                      c='red', label='Source (Reference)', s=2, alpha=0.7)
            
            # Plot target points in blue
            ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                      c='blue', label='Target (Subject)', s=2, alpha=0.7)
            
            ax.set_title(f'ALPACA Debug: {title}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            
            # Make all axes equal scale for better visualization
            ax.set_box_aspect([1,1,1])
            
            # Set axis labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add grid for better reference
            ax.grid(True, alpha=0.3)
            
            # Save with high quality
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            print(f"Debug image saved: {output_path}")
            
        except Exception as e:
            print(f"Warning: Could not save debug image '{title}': {str(e)}")

    def plot_debug_step_ransac_exclusions(
        self,
        source_points_transformed,
        target_points,
        included_mask,
        title,
        output_dir=None,
    ):
        """
        Same as plot_debug_step but source split: included (red), excluded by scoring mask (green).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f"debug_{title.replace(' ', '_')}.jpg"
                )
            else:
                output_path = f"debug_{title.replace(' ', '_')}.jpg"

            inc = np.asarray(included_mask, dtype=bool)
            sin = source_points_transformed[inc]
            sex = source_points_transformed[~inc]

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection="3d")
            if len(sin) > 0:
                ax.scatter(
                    sin[:, 0],
                    sin[:, 1],
                    sin[:, 2],
                    c="red",
                    label="Source included (ICP will use)",
                    s=2,
                    alpha=0.7,
                )
            if len(sex) > 0:
                ax.scatter(
                    sex[:, 0],
                    sex[:, 1],
                    sex[:, 2],
                    c="green",
                    label="Source excluded (Otsu tail mask)",
                    s=2,
                    alpha=0.7,
                )
            ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                c="blue",
                label="Target (Subject)",
                s=2,
                alpha=0.7,
            )
            ax.set_title(f"ALPACA Debug: {title}", fontsize=14, fontweight="bold")
            ax.legend(fontsize=12)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.grid(True, alpha=0.3)
            plt.savefig(
                output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )
            plt.close(fig)
            print(f"Debug image saved: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save exclusion debug image '{title}': {str(e)}")

    def pca_pre_alignment(self, source_points, target_points):
        """
        Performs PCA-based pre-alignment to align principal axes using ONLY proper rotations.
        This provides a better starting point for RANSAC by aligning the main
        orientations of both point clouds, reducing the search space significantly.
        
        IMPORTANT: Only uses proper rotations (det = +1) to preserve anatomical validity.
        No reflections are used - only rotations between eigenvector bases.
        
        Args:
            source_points: np.array (N, 3) source landmarks
            target_points: np.array (M, 3) target landmarks
            
        Returns:
            transformation_matrix: 4x4 proper rotation transformation to pre-align target to source
        """
        # Center both point clouds
        source_centered = source_points - np.mean(source_points, axis=0)
        target_centered = target_points - np.mean(target_points, axis=0)
        
        # Compute PCA for both point clouds
        source_cov = np.cov(source_centered.T)
        target_cov = np.cov(target_centered.T)
        
        source_eigenvals, source_eigenvecs = np.linalg.eigh(source_cov)
        target_eigenvals, target_eigenvecs = np.linalg.eigh(target_cov)
        
        # Sort by eigenvalues (descending order)
        source_idx = np.argsort(source_eigenvals)[::-1]
        target_idx = np.argsort(target_eigenvals)[::-1]
        
        source_eigenvecs = source_eigenvecs[:, source_idx]
        target_eigenvecs = target_eigenvecs[:, target_idx]
        
        # Ensure right-handed coordinate systems
        source_eigenvecs = self._ensure_right_handed(source_eigenvecs)
        target_eigenvecs = self._ensure_right_handed(target_eigenvecs)
        
        # Test different axis alignments to find best PROPER rotation
        best_rotation = None
        best_score = float('inf')
        
        # Try all possible axis alignments (6 permutations)
        axis_permutations = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], 
            [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ]
        
        for perm in axis_permutations:
            # Create permuted target eigenvectors
            test_eigenvecs = target_eigenvecs[:, perm].copy()
            
            # Compute rotation matrix
            rotation = source_eigenvecs @ test_eigenvecs.T
            
            # CRITICAL: Ensure this is a proper rotation (det = +1)
            # If det = -1, we have a reflection - fix by flipping one eigenvector
            if np.linalg.det(rotation) < 0:
                # Flip the last eigenvector to make it a proper rotation
                test_eigenvecs[:, 2] *= -1
                rotation = source_eigenvecs @ test_eigenvecs.T
                
            # Verify we now have a proper rotation
            det_rotation = np.linalg.det(rotation)
            if abs(det_rotation - 1.0) > 1e-10:
                print(f"Warning: rotation determinant = {det_rotation}, should be 1.0")
                continue
                
            # Test quality by measuring alignment of principal directions
            aligned_target = (rotation @ target_centered.T).T
            score = np.sum(np.abs(np.std(aligned_target, axis=0) - np.std(source_centered, axis=0)))
            
            if score < best_score:
                best_score = score
                best_rotation = rotation
        
        # Create 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = best_rotation
        
        # Verify final transformation is proper rotation
        final_det = np.linalg.det(best_rotation)
        print(f"PCA pre-alignment completed with proper rotation (det = {final_det:.6f}, score = {best_score:.6f})")
        
        if abs(final_det - 1.0) > 1e-6:
            print(f"ERROR: Final transformation is not a proper rotation! det = {final_det}")
            
        return transformation
    
    def _ensure_right_handed(self, eigenvecs):
        """Ensure eigenvector matrix forms a right-handed coordinate system"""
        if np.linalg.det(eigenvecs) < 0:
            eigenvecs[:, 2] *= -1  # Flip the third eigenvector
        return eigenvecs
    
    def test_axis_rotations(self, source_pcd, target_pcd, initial_transformation, description=""):
        """
        Tests 180-degree rotations around each axis to handle symmetry issues.
        This addresses the common problem where anatomically symmetric structures
        can result in upside-down alignments that are mathematically valid but 
        anatomically incorrect.
        
        IMPORTANT: These are pure 180° rotations, NOT reflections. This preserves
        the anatomical validity while testing different orientations that could
        arise from symmetry in the alignment process.
        
        Mathematical note: The group of 180° rotations around coordinate axes
        contains exactly 4 distinct elements: {I, R_x, R_y, R_z}. All combinations
        of these rotations reduce to one of these four basic rotations.
        
        Args:
            source_pcd: source point cloud
            target_pcd: target point cloud  
            initial_transformation: 4x4 transformation matrix to test rotations on
            description: string for debug output
            
        Returns:
            best_transformation: 4x4 transformation matrix with best rotation applied
        """
        print(f"Testing 180° axis rotations for {description}...")
        
        # Create rotation matrices for 180-degree rotations around each axis
        # These are proper rotations, NOT reflections - anatomically safe
        rot_x_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        rot_y_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  
        rot_z_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # Test only the 4 distinct 180° rotations (combinations are redundant)
        transformations_to_test = [
            ("original", initial_transformation),
            ("rot_x_180", rot_x_180 @ initial_transformation),
            ("rot_y_180", rot_y_180 @ initial_transformation),
            ("rot_z_180", rot_z_180 @ initial_transformation)
        ]

        for name, transform in transformations_to_test:
            print(f"{name}:")
            for i in range(4):
                row = transform[i, :]
                print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        
        best_transformation = initial_transformation
        best_score = -1
        best_name = "original"
        
        # Check if both point clouds have normals
        if not (source_pcd.has_normals() and target_pcd.has_normals()):
            print("  Warning: Point clouds missing normals. Returning original transformation.")
            return best_transformation
        
        target_tree = o3d.geometry.KDTreeFlann(target_pcd)
        target_normals = np.asarray(target_pcd.normals)
        
        for name, test_transform in transformations_to_test:
            # Transform source and compute normal consistency
            source_test = copy.deepcopy(source_pcd)
            source_test.transform(test_transform)
            source_normals = np.asarray(source_test.normals)
            
            # Calculate normal consistency
            total_normal_alignment = 0
            valid_points = 0
            
            for i, point in enumerate(np.asarray(source_test.points)):
                [_, idx, _] = target_tree.search_knn_vector_3d(point, 1)
                if len(idx) > 0:
                    normal_alignment = abs(np.dot(source_normals[i], target_normals[idx[0]]))
                    total_normal_alignment += normal_alignment
                    valid_points += 1
            
            if valid_points > 0:
                normal_consistency = total_normal_alignment / valid_points
            else:
                normal_consistency = 0.0
            
            print(f"  {name}: normal_consistency={normal_consistency:.4f}")
            
            if normal_consistency > best_score:
                best_score = normal_consistency
                best_transformation = test_transform
                best_name = name
        
        print(f"Best rotation: {best_name} with normal consistency: {best_score:.4f}")
        return best_transformation

    def _align_landmarks_to_mesh_impl(
        self,
        source_vertices,
        source_faces,
        target_vertices,
        target_faces,
        target_metadata_path,
        scaling_mode=False,
        outer_surface=False,
        source_landmarks=None,
        update_alignment_metadata=True,
        debug_subdir='debug_alignment',
        ransac_only=False,
        target_poisson_points=None,
    ):

        # PCA pre-alignment rotates the target's principal axes onto the source's before RANSAC.
        # Disabled by default: FPFH is rotation-invariant so PCA is not needed, and on
        # bilaterally symmetric shapes PCA sign ambiguity can introduce a 180° flip that
        # test_axis_rotations then has to correct.  Set to True to re-enable for comparison.
        do_pca = False

        # source is the reference
        # target is the subject

        # save a copy of the original target and source vertices so that we can plot the final transformation applied to it
        target_vertices_original = target_vertices.copy()
        if source_landmarks is None:
            source_vertices_original = source_vertices.copy()
        else:
            source_vertices_original = None

        # Extract scan directory from metadata path for debug images
        scan_dir = os.path.dirname(target_metadata_path)
        debug_dir = os.path.join(scan_dir, debug_subdir)
        print(f"Debug images will be saved to: {debug_dir}")

        # Validate RANSAC result (diagnostic; distances scale with normalization — use global_peak_threshold).
        def validate_alignment(
            source_pcd,
            target_pcd,
            transformation,
            random_params,
            attempt_number,
            global_peak_threshold=None,
            include_mask=None,
            quiet=False,
        ):
            """Returns (avg_distance, normal_consistency, passed_heuristic_gates).

            If include_mask is a length-N boolean array (N = source points), only those
            True indices contribute to avg_distance and normal_consistency — matches the
            burn-in / ICP subset used for median_nn so the alignment gate is not diluted
            by excluded tail points (e.g. embryo vs reference anatomy mismatch).
            """
            source_transformed = copy.deepcopy(source_pcd)
            source_transformed.transform(transformation)

            source_points = np.asarray(source_transformed.points)
            source_normals = np.asarray(source_transformed.normals)
            target_normals = np.asarray(target_pcd.normals)

            target_tree = o3d.geometry.KDTreeFlann(target_pcd)

            total_points = len(source_points)
            if include_mask is not None:
                mask_arr = np.asarray(include_mask, dtype=bool).ravel()
                if mask_arr.shape[0] != total_points:
                    raise ValueError(
                        "include_mask length must match source point count "
                        f"({mask_arr.shape[0]} != {total_points})"
                    )
                indices = np.nonzero(mask_arr)[0]
                if indices.size == 0:
                    indices = np.arange(total_points, dtype=np.int64)
            else:
                indices = np.arange(total_points, dtype=np.int64)

            n_use = int(indices.size)
            good_alignments = 0
            avg_distance = 0.0

            for i in indices:
                point = source_points[i]
                normal = source_normals[i]
                [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
                avg_distance += np.sqrt(dist[0])
                normal_alignment = abs(np.dot(normal, target_normals[idx[0]]))
                if normal_alignment > 0.7:
                    good_alignments += 1

            avg_distance /= max(n_use, 1)
            normal_consistency = good_alignments / max(n_use, 1)

            if global_peak_threshold is not None and float(global_peak_threshold) > 0:
                distance_threshold = 0.5 * float(global_peak_threshold)
            else:
                distance_threshold = 0.01
            if attempt_number <= 10:
                consistency_threshold = 0.9
            else:
                consistency_threshold = 0.75

            if not quiet:
                print("Validation of RANSAC results:")
                print(f"Average distance: {avg_distance:.4f}")
                print(f"Normal consistency: {normal_consistency:.4f}")
                print(
                    f"Heuristic gates: avg_dist < {distance_threshold:.6f} "
                    f"(0.5× Otsu peak/tail), consistency > {consistency_threshold}"
                )

            if random_params:
                random_params["average_distance"] = avg_distance
                random_params["normal_consistency"] = normal_consistency

            passed = (avg_distance < distance_threshold) and (
                normal_consistency > consistency_threshold
            )
            return avg_distance, normal_consistency, passed

        def otsu_threshold_1d(data, n_bins=256):
            """
            Otsu threshold on 1D positive distances: separates peak (near match)
            from tail (no match). Returns threshold in same units as data.
            """
            d = np.asarray(data, dtype=np.float64).ravel()
            d = d[np.isfinite(d)]
            if d.size < 50:
                return float(np.median(d)) if d.size else 1e-3
            hist, bin_edges = np.histogram(d, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
            total = hist.sum()
            if total <= 0:
                return float(np.median(d))
            prob = hist.astype(np.float64) / total
            omega = np.cumsum(prob)
            mu = np.cumsum(prob * bin_centers)
            mu_t = mu[-1]
            denom = omega * (1.0 - omega) + 1e-12
            sigma_b_sq = (mu_t * omega - mu) ** 2 / denom
            sigma_b_sq[~np.isfinite(sigma_b_sq)] = 0.0
            idx = int(np.argmax(sigma_b_sq))
            t = float(bin_centers[idx])
            # Keep threshold inside bulk of data (avoid degenerate minima at extremes).
            p10, p90 = np.percentile(d, [10, 90])
            t = float(np.clip(t, max(p10 * 0.25, 1e-7), p90))
            return max(t, 1e-7)

        def median_iqr_included(distances, included_mask):
            """
            Consensus surface match: median NN distance over included points (lower = better).
            IQR of those distances (lower = tighter peak, tie-break when medians nearly equal).
            """
            d = np.asarray(distances, dtype=np.float64)[included_mask]
            if d.size == 0:
                return float("inf"), float("inf")
            med = float(np.median(d))
            q25, q75 = np.percentile(d, [25.0, 75.0])
            iqr = float(q75 - q25)
            return med, iqr

        def collect_nn_distances_for_transform(transformation):
            """Per-source-point NN distances after applying transformation (full-res clouds; same order as source_pcd_full)."""
            source_tmp = copy.deepcopy(source_pcd_full)
            source_tmp.transform(transformation)
            target_tree = o3d.geometry.KDTreeFlann(target_pcd_full)
            out = np.empty(len(np.asarray(source_tmp.points)), dtype=np.float64)
            for i, pt in enumerate(np.asarray(source_tmp.points)):
                [_, _, dist_sq] = target_tree.search_knn_vector_3d(pt, 1)
                out[i] = np.sqrt(dist_sq[0])
            return out

        def save_global_burnin_histogram(flat_distances, global_threshold, output_dir):
            """Histogram of all burn-in NN distances with Otsu peak/tail split line."""
            try:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "debug_Global_burnin_NN_distances.jpg")
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.hist(
                    flat_distances,
                    bins=100,
                    edgecolor="black",
                    alpha=0.7,
                    color="steelblue",
                )
                ax.axvline(
                    global_threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Otsu global_threshold = {global_threshold:.5f}",
                )
                n_below = int(np.sum(flat_distances < global_threshold))
                ax.set_title(
                    f"All burn-in NN distances (stacked poses)\n"
                    f"{n_below}/{len(flat_distances)} below threshold "
                    f"({100.0 * n_below / len(flat_distances):.1f}%)"
                )
                ax.set_xlabel("NN distance (normalized frame)")
                ax.set_ylabel("Count")
                ax.legend()
                fig.tight_layout()
                plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"Global burn-in histogram saved: {path}")
            except Exception as e:
                print(f"Warning: could not save global burn-in histogram: {e}")

        def save_nn_distance_histogram(
            distances, global_threshold, title, output_dir, subtitle_extra=""
        ):
            try:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(
                    output_dir, f"debug_{title.replace(' ', '_')}.jpg"
                )
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.hist(distances, bins=80, edgecolor="black", alpha=0.7, color="steelblue")
                ax.axvline(
                    global_threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"global_peak_threshold = {global_threshold:.5f}",
                )
                n_in = int(np.sum(distances < global_threshold))
                sub = (
                    f"{n_in}/{len(distances)} below threshold "
                    f"({100.0 * n_in / len(distances):.1f}%)"
                )
                if subtitle_extra:
                    sub = subtitle_extra + "\n" + sub
                ax.set_title(f"{title}\n{sub}")
                ax.set_xlabel("NN distance (normalized frame)")
                ax.set_ylabel("Count")
                ax.legend()
                fig.tight_layout()
                plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"Distance histogram saved: {path}")
            except Exception as e:
                print(f"Warning: could not save distance histogram: {e}")

        def filter_source_pcd_for_icp(
            source_pcd,
            target_pcd,
            init_transform,
            nn_threshold,
            label,
            source_keep_mask=None,
        ):
            """
            Drop source points whose NN distance to target (after init_transform) exceeds
            nn_threshold. If source_keep_mask is given (bool length n), only those indices
            may be kept (burn-in consensus mask).
            """
            n = len(source_pcd.points)
            if n == 0:
                return copy.deepcopy(source_pcd)
            if source_keep_mask is not None:
                sk = np.asarray(source_keep_mask, dtype=bool)
                if sk.shape[0] != n:
                    print(
                        f"Warning: ICP filter '{label}': mask length {sk.shape[0]} != n {n}; "
                        f"ignoring mask."
                    )
                    source_keep_mask = None
            src_w = copy.deepcopy(source_pcd)
            src_w.transform(init_transform)
            tree = o3d.geometry.KDTreeFlann(target_pcd)
            pts_w = np.asarray(src_w.points)
            keep = []
            for i in range(n):
                if source_keep_mask is not None and not source_keep_mask[i]:
                    continue
                [_, _, d2] = tree.search_knn_vector_3d(pts_w[i], 1)
                if np.sqrt(d2[0]) < nn_threshold:
                    keep.append(i)
            min_keep = min(200, max(30, n // 20))
            if len(keep) < min_keep:
                # Fall back: all burn-in-included points without NN cut (better than full cloud).
                if source_keep_mask is not None:
                    fallback = np.flatnonzero(source_keep_mask).tolist()
                    if len(fallback) >= min_keep:
                        print(
                            f"Warning: ICP filter '{label}': NN gate left {len(keep)}/{n} pts; "
                            f"using {len(fallback)} burn-in-included pts without NN gate."
                        )
                        out = source_pcd.select_by_index(fallback)
                        return out
                print(
                    f"Warning: ICP filter '{label}' kept only {len(keep)}/{n} points "
                    f"(minimum {min_keep} required); using unfiltered source."
                )
                return copy.deepcopy(source_pcd)
            out = source_pcd.select_by_index(keep)
            mask_note = " + burn-in mask" if source_keep_mask is not None else ""
            print(
                f"ICP source filter '{label}': kept {len(keep)}/{n} points "
                f"(NN < {nn_threshold:.6f}){mask_note}."
            )
            return out

        # Single frame for all tries: one Poisson build, then N RANSAC variants (seed + params).
        # Rebuilding mid-loop would change normalization and make stored transforms incomparable.
        n_ransac_attempts = 40
        max_retries = n_ransac_attempts
        attempt = 0
        burn_in_distances = None
        burn_in_idx = 0
        candidate_transforms = []
        global_peak_threshold = 0.005

        if outer_surface:
            if source_landmarks is None:
                print("[outer surface] source vertices before: ", source_vertices.shape)
                full_source_vertices, full_source_faces = source_vertices, source_faces
                source_vertices, source_faces = self.get_outer_mesh(source_vertices, source_faces)
                if len(source_vertices) == 0 or len(source_faces) == 0:
                    print(
                        "Warning: outer-surface extraction emptied the reference mesh; "
                        "falling back to the full mesh."
                    )
                    source_vertices, source_faces = full_source_vertices, full_source_faces
                # Save a ply with the outer surface to debug
                # o3d.io.write_triangle_mesh("outer_surface_reference.ply", o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(source_vertices), triangles=o3d.utility.Vector3iVector(source_faces)))
                print("[outer surface] source vertices after: ", source_vertices.shape)

            print("[outer surface] target vertices before: ", target_vertices.shape)
            full_target_vertices, full_target_faces = target_vertices, target_faces
            target_vertices, target_faces = self.get_outer_mesh(target_vertices, target_faces)
            if len(target_vertices) == 0 or len(target_faces) == 0:
                print(
                    "Warning: outer-surface extraction emptied the subject mesh; "
                    "falling back to the full mesh."
                )
                target_vertices, target_faces = full_target_vertices, full_target_faces
            # Save a ply with the outer surface to debug
            # o3d.io.write_triangle_mesh("outer_surface_target.ply", o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(target_vertices), triangles=o3d.utility.Vector3iVector(target_faces)))
            print("[outer surface] target vertices after: ", target_vertices.shape)
            

        if scaling_mode:
            # aka reference (upload-fit uses landmark points when source_vertices is None)
            if source_landmarks is not None:
                source_points_for_scale = np.asarray(source_landmarks, dtype=np.float64)
            else:
                source_points_for_scale = source_vertices
            source_centroid = np.mean(source_points_for_scale, axis=0)
            source_mean_distance_from_centroid = np.mean(
                np.linalg.norm(source_points_for_scale - source_centroid, axis=1)
            )
            print("ref_mean_distance_from_centroid: ", source_mean_distance_from_centroid)
            # aka subject
            target_centroid = np.mean(target_vertices, axis=0)
            target_mean_distance_from_centroid = np.mean(np.linalg.norm(target_vertices - target_centroid, axis=1))
            print("scan_mean_distance_from_centroid: ", target_mean_distance_from_centroid)
            scale_match = source_mean_distance_from_centroid / target_mean_distance_from_centroid
            target_vertices = target_vertices * scale_match
            print("scale to match data to reference: ", scale_match)
        else:
            scale_match = 1

        # Vertex ratio (subject vs reference): drives subject Poisson count relative to fixed reference budget.
        # Upload-fit uses landmark points as source; Poisson budget may be raised for denser RANSAC.
        if source_landmarks is not None:
            if target_poisson_points is not None:
                n_landmarks_target = int(target_poisson_points)
            else:
                n_landmarks_target = ALIGNMENT_REFERENCE_POISSON_POINTS
            print(
                f"Poisson landmark budget (upload fit): source={len(np.asarray(source_landmarks))} "
                f"landmarks, target={n_landmarks_target} mesh samples"
            )
        else:
            source_count = len(source_vertices)
            vertex_ratio = len(target_vertices) / max(source_count, 1)
            print("vertex_ratio: ", vertex_ratio)
            n_landmarks_target = max(
                1, int(round(ALIGNMENT_REFERENCE_POISSON_POINTS * vertex_ratio))
            )
            print(
                f"Poisson landmark budget: reference={ALIGNMENT_REFERENCE_POISSON_POINTS}, "
                f"subject={n_landmarks_target} (ratio {vertex_ratio:.6f})"
            )

        if source_landmarks is not None:
            source_landmarks = np.asarray(source_landmarks, dtype=np.float64)
            print(f"Using {len(source_landmarks)} uploaded landmarks as source point cloud")
        else:
            source_landmarks = self.create_landmarks_from_mesh(
                source_vertices, source_faces, n_landmarks=ALIGNMENT_REFERENCE_POISSON_POINTS
            )
                
        
        while attempt < max_retries:

            # Phase 1: FPFH + RANSAC with low resolution
            gc.collect()

            # Mix wall time with attempt index so rapid loop iterations never share the same
            # seed (same millisecond would otherwise repeat random_params and correspondence draws).
            current_seed = (
                int(time.time() * 1000) + attempt * 1103515245
            ) % (2**32 - 1)
            np.random.seed(current_seed)
            random_state = np.random.RandomState(current_seed)

            # Single full rebuild at loop start only — keeps one normalized frame so all
            # n_ransac_attempts candidate transforms and distance columns are comparable.
            if attempt == 0:
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_landmarks)
                
                target_mesh = o3d.geometry.TriangleMesh()
                target_mesh.vertices = o3d.utility.Vector3dVector(target_vertices)
                target_mesh.triangles = o3d.utility.Vector3iVector(target_faces)                

                # Normals are required before Poisson-disk sampling can distribute points evenly.
                target_mesh.compute_vertex_normals()
                n_landmarks = n_landmarks_target
                print(f"DEBUG: target_mesh has {len(target_mesh.vertices)} vertices.")
                print(f"DEBUG: n_landmarks (subject): {n_landmarks}")

                print(f"DEBUG: source_landmarks has {len(source_landmarks)} points.")
                # Build three resolution tiers for each cloud.
                # Full resolution is used for FPFH + RANSAC (dense neighborhoods, stable descriptors).
                # Low (1/5), medium (1/3), and full are used for coarse-to-fine ICP after RANSAC.
                source_pcd_low = o3d.geometry.PointCloud()
                source_pcd_med = o3d.geometry.PointCloud()
                source_pcd_full = o3d.geometry.PointCloud()
                
                # Uniformly spaced index strides give even spatial coverage at each resolution.
                n_points_low = max(1, len(source_landmarks) // 5)
                n_points_med = max(1, len(source_landmarks) // 3)
                source_indices_low = np.linspace(0, len(source_landmarks)-1, n_points_low, dtype=int)
                source_indices_med = np.linspace(0, len(source_landmarks)-1, n_points_med, dtype=int)
                
                source_pcd_low.points = o3d.utility.Vector3dVector(source_landmarks[source_indices_low])
                source_pcd_med.points = o3d.utility.Vector3dVector(source_landmarks[source_indices_med])
                source_pcd_full.points = o3d.utility.Vector3dVector(source_landmarks)
                
                n_tgt_low = max(1, n_landmarks // 5)
                n_tgt_med = max(1, n_landmarks // 3)
                target_pcd_low = target_mesh.sample_points_poisson_disk(number_of_points=n_tgt_low)
                target_pcd_med = target_mesh.sample_points_poisson_disk(number_of_points=n_tgt_med)
                target_pcd_full = target_mesh.sample_points_poisson_disk(number_of_points=n_landmarks)
                
                self.plot_debug_step(np.asarray(source_pcd_low.points), 
                                    np.asarray(target_pcd_low.points), 
                                    "Initial Point Clouds", debug_dir)
                
                # Compute centroids from the same full-res clouds used for RANSAC/ICP.
                source_centroid = np.mean(source_landmarks, axis=0)
                target_centroid = np.mean(np.asarray(target_pcd_full.points), axis=0)
                
                source_pcd_full.translate(-source_centroid)
                target_pcd_full.translate(-target_centroid)
                
                # RMS distance from centroid (N-independent). The Frobenius norm ||P||_F equals
                # sqrt(N) * RMS, so using Frobenius shrinks the normalized cloud when N grows and
                # breaks fixed FPFH/RANSAC radii. RMS keeps the same spatial extent for any sample size.
                _pts_c = np.asarray(source_pcd_full.points)
                scale_factor = float(np.sqrt(np.mean(np.sum(_pts_c * _pts_c, axis=1))))
                scale_factor = max(scale_factor, 1e-15)
                center = np.zeros((3, 1))
                
                source_pcd_full.scale(1 / scale_factor, center)
                target_pcd_full.scale(1 / scale_factor, center)
                
                # Apply the same centring and scale to the coarser clouds so all resolution
                # tiers share a common coordinate frame after this block.
                source_pcd_med.translate(-source_centroid)
                source_pcd_low.translate(-source_centroid)
                target_pcd_med.translate(-target_centroid)
                target_pcd_low.translate(-target_centroid)
                
                source_pcd_med.scale(1 / scale_factor, center)
                source_pcd_low.scale(1 / scale_factor, center)
                target_pcd_med.scale(1 / scale_factor, center)
                target_pcd_low.scale(1 / scale_factor, center)
                
                self.plot_debug_step(np.asarray(source_pcd_low.points), 
                                np.asarray(target_pcd_low.points), 
                                "01_After_Centering_and_Normalization", debug_dir)

                # ====== ADAPTIVE GEOMETRY-BASED FEATURE SCALE ESTIMATION ======
                # Reference-only diagonal (source_pcd_full after RMS normalization) drives scales so
                # all subjects share the same feature radii relative to the reference anatomy.
                #
                # FPFH radius is a fraction of that diagonal from measured smoothness: smooth surfaces
                # need *smaller* radii so descriptors stay discriminative (large radii collapse mutual
                # matching). Rough surfaces use a moderately larger fraction to average noise.
                
                _pts_norm = np.asarray(source_pcd_full.points)
                _cloud_diagonal = float(np.linalg.norm(_pts_norm.max(axis=0) - _pts_norm.min(axis=0)))
                n_source_pts = int(len(_pts_norm))
                n_source_pts = max(n_source_pts, 1)

                # Fixed 4% for normals: enough to estimate stable tangent frames for spread measurement.
                normal_max_nn = max(30, int(round(0.04 * n_source_pts)))
                normal_radius_spread = _cloud_diagonal * float(
                    np.sqrt(normal_max_nn / n_source_pts)
                )

                # Compute surface smoothness by sampling ~200 points and measuring normal angular spread.
                # For each sample: find 30 neighbors, compute angles between center normal and neighbor normals,
                # take mean angle → measure of curvature/variation at that location.
                _src_geo = copy.deepcopy(source_pcd_full)
                _src_geo.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=normal_radius_spread,
                        max_nn=normal_max_nn,
                    )
                )
                _src_geo.orient_normals_consistent_tangent_plane(100)
                _pts_geo = np.asarray(_src_geo.points)
                _nrm_geo = np.asarray(_src_geo.normals)
                _k_spread = min(31, n_source_pts)
                median_spread = 8.0  # Default (degrees); will be overwritten if enough points
                if _k_spread >= 2:
                    _tree_geo = o3d.geometry.KDTreeFlann(_src_geo)
                    n_sample = min(200, n_source_pts)
                    sample_idx = np.linspace(0, n_source_pts - 1, n_sample, dtype=np.int64)
                    spreads = []
                    for qi in sample_idx:
                        _, idx, _ = _tree_geo.search_knn_vector_3d(_pts_geo[qi], _k_spread)
                        idx = np.asarray(idx, dtype=np.int64).ravel()[:_k_spread]
                        n_i = _nrm_geo[qi]
                        nj = _nrm_geo[idx[1:]]  # Exclude self
                        if nj.size == 0:
                            continue
                        dots = np.clip(np.abs(np.dot(nj, n_i)), 0.0, 1.0)
                        spreads.append(float(np.mean(np.degrees(np.arccos(dots)))))
                    if spreads:
                        median_spread = float(np.median(spreads))

                # Map median normal spread (degrees) → FPFH radius as fraction of reference diagonal.
                #   5° (smooth): 8%  — local enough for discriminative FPFH on smooth bone
                #  28° (rough): 15% — larger context to stabilize descriptors on noisy geometry
                spread_low, spread_high = 5.0, 28.0
                frac_smooth, frac_complex = 0.08, 0.15
                t = (median_spread - spread_low) / (spread_high - spread_low)
                t = float(np.clip(t, 0.0, 1.0))
                fpfh_frac = frac_smooth * (1.0 - t) + frac_complex * t
                fpfh_radius_base = _cloud_diagonal * fpfh_frac

                # Cap neighbor count for FPFH; spatial scale is controlled by fpfh_radius_base.
                fpfh_max_nn = max(30, min(300, n_source_pts))

                # RANSAC normal search: same max_nn, radius slightly tighter than FPFH when possible.
                normal_radius_base = min(normal_radius_spread, fpfh_radius_base * 0.8)

                print(
                    f"DEBUG: cloud_diagonal={_cloud_diagonal:.4f}, n_source_pts={n_source_pts}, "
                    f"median_normal_spread_deg={median_spread:.2f}, fpfh_frac={fpfh_frac:.4f}, "
                    f"fpfh_max_nn={fpfh_max_nn}, fpfh_radius_base={fpfh_radius_base:.4f}, "
                    f"normal_max_nn={normal_max_nn}, normal_radius_base={normal_radius_base:.4f}"
                )

                n_ransac_pts = len(source_pcd_full.points)
                if burn_in_distances is None:
                    burn_in_distances = np.zeros((n_ransac_pts, n_ransac_attempts))
                elif burn_in_distances.shape[0] != n_ransac_pts:
                    print(
                        "Warning: full-res source point count changed after rebuild; "
                        "resetting statistics buffer."
                    )
                    burn_in_distances = np.zeros((n_ransac_pts, n_ransac_attempts))
                    burn_in_idx = 0
                    candidate_transforms.clear()

            
            attempt += 1
            print(f"\nAttempting RANSAC alignment: try {attempt}/{max_retries} with seed {current_seed}")

            print(f"DEBUG: RANSAC uses full-res — source_pcd_full {len(source_pcd_full.points)} pts, target_pcd_full {len(target_pcd_full.points)} pts.")

            # Every 10th try (1, 11, 21, …): wider parameter jitter for diversity without
            # rebuilding Poisson clouds (same normalized frame for all candidates).
            wide_decile = ((attempt - 1) % 10 == 0)
            if wide_decile:
                print(
                    f"Try {attempt}/{max_retries}: wide parameter block "
                    f"(broader FPFH / RANSAC / normal search jitter)."
                )
            # Jitter around geometry-derived bases so different RANSAC attempts explore nearby
            # scales. RANSAC inlier distance is a fraction of the (jittered) FPFH radius.
            fpfh_jitter = random_state.uniform(0.7, 1.4) if wide_decile else random_state.uniform(0.85, 1.2)
            normal_jitter = random_state.uniform(0.7, 1.4) if wide_decile else random_state.uniform(0.85, 1.2)
            ransac_dist = fpfh_radius_base * fpfh_jitter * (
                random_state.uniform(0.08, 0.20) if wide_decile else random_state.uniform(0.10, 0.15)
            )

            random_params = {
                "normal_search_radius": normal_radius_base * normal_jitter,
                "max_nn_normals": normal_max_nn,
                "fpfh_radius": fpfh_radius_base * fpfh_jitter,
                "max_nn_fpfh": fpfh_max_nn,
                "ransac_max_correspondence_distance": ransac_dist,
                "ransac_n": 3,
                "normal_consistency": 0.0,
                "average_distance": 0.0
            }


            # Normals on full-res clouds: dense neighborhoods → stable FPFH (Porto-style single resolution for RANSAC).
            try:
                source_pcd_full.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["normal_search_radius"], 
                        max_nn=random_params["max_nn_normals"]))
                target_pcd_full.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["normal_search_radius"], 
                        max_nn=random_params["max_nn_normals"]))
            except Exception as e:
                print(f"Error estimating normals: {str(e)}")

            # Propagate a globally consistent normal orientation using the tangent-plane
            # graph.  orient_normals_consistent_tangent_plane has no seed parameter, so
            # results are deterministic given the point positions and k=100 neighbours.
            source_pcd_full.orient_normals_consistent_tangent_plane(100)
            target_pcd_full.orient_normals_consistent_tangent_plane(100)

            # Single-scale FPFH at a radius proportional to the cloud diagonal (Porto-style).
            # A semi-global radius (5 × voxel_size) captures enough shape context to distinguish
            # points on smooth anatomy (e.g., embryo heads) where purely local radii produce
            # identical descriptors everywhere and break correspondence matching.
            try:
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_pcd_full,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["fpfh_radius"],
                        max_nn=random_params["max_nn_fpfh"]))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_pcd_full,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["fpfh_radius"],
                        max_nn=random_params["max_nn_fpfh"]))
            except Exception as e:
                print(f"Error computing FPFH features: {str(e)}")
                source_fpfh = None
                target_fpfh = None

            # Porto-style RANSAC: pass FPFH Feature objects directly and let Open3D handle
            # correspondence sampling internally.  mutual_filter=True enforces mutual nearest
            # neighbour (eliminates ambiguous matches without the over-aggressive ratio test).
            # EdgeLength checker adds geometric consistency: the ratio of matched edge lengths
            # must agree to within 10 %, which reliably rejects incorrect correspondences even
            # on smooth surfaces where descriptor distances alone are uninformative.
            best_ransac_result = None
            if source_fpfh is not None and target_fpfh is not None:
                try:
                    best_ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_pcd_full, target_pcd_full,
                        source_fpfh, target_fpfh,
                        mutual_filter=True,
                        max_correspondence_distance=random_params["ransac_max_correspondence_distance"],
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                        ransac_n=random_params["ransac_n"],
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                random_params["ransac_max_correspondence_distance"]),
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            max_iteration=int(self.params["max_ransac_iter"]),
                            confidence=float(self.params.get("ransac_confidence", 0.9999)),
                        )
                    )
                    print(f"DEBUG: current_ransac.fitness: {best_ransac_result.fitness}")
                except Exception as e:
                    print(f"Error during RANSAC: {str(e)}")

            if best_ransac_result is None:
                print("Warning: RANSAC failed. Using identity transformation as fallback.")
                best_ransac_result = type('', (), {})()
                best_ransac_result.transformation = np.eye(4)
                best_ransac_result.fitness = 0.0
                best_ransac_result.inlier_rmse = float('inf')
                
            ransac_result = best_ransac_result
            
            # AXIS FLIP TESTING: Test flipped orientations to handle symmetry issues
            # This addresses cases where the scan gets aligned upside-down due to symmetry
            print("Testing axis rotations for RANSAC result...")
            ransac_result.transformation = self.test_axis_rotations(
                source_pcd_full, target_pcd_full, 
                ransac_result.transformation, 
                "RANSAC result"
            )

            T = ransac_result.transformation.copy()
            candidate_transforms.append(T)
            dist_col = collect_nn_distances_for_transform(T)
            burn_in_distances[:, burn_in_idx] = dist_col
            burn_in_idx += 1
            print(
                f"Recorded candidate {attempt}/{n_ransac_attempts} "
                f"(distance column {burn_in_idx}/{n_ransac_attempts})."
            )

        print(
            f"\nScoring {len(candidate_transforms)} candidates: Otsu threshold on all burn-in "
            f"NN distances; winner from composite score; ICP/debug mask = top-10 tries, "
            f"exclude if in Otsu tail ≥90% of those tries."
        )
        flat_burn = burn_in_distances.ravel()
        global_peak_threshold = otsu_threshold_1d(flat_burn)
        print(
            f"  Global Otsu threshold (peak vs tail on all burn-in distances): "
            f"{global_peak_threshold:.6f}"
        )
        save_global_burnin_histogram(flat_burn, global_peak_threshold, debug_dir)

        n_pts_burn = burn_in_distances.shape[0]
        score_all_included = np.ones(n_pts_burn, dtype=bool)

        medians = []
        iqrs = []
        normal_consistencies = []

        # Composite scoring: median / IQR over all source points (no mask) so ranking is stable.
        for i in range(len(candidate_transforms)):
            m, iq = median_iqr_included(burn_in_distances[:, i], score_all_included)
            medians.append(m)
            iqrs.append(iq)
            
            _, nc, _ = validate_alignment(
                source_pcd_full,
                target_pcd_full,
                candidate_transforms[i],
                {},
                attempt,
                global_peak_threshold=global_peak_threshold,
            )
            normal_consistencies.append(nc)

        medians = np.asarray(medians, dtype=np.float64)
        iqrs = np.asarray(iqrs, dtype=np.float64)
        normal_consistencies = np.asarray(normal_consistencies, dtype=np.float64)
        
        # Composite score: lower median, tighter IQR, higher normal consistency.
        # score = median * (1 + IQR) / nc; if nc near zero, clamp to avoid division issues.
        USE_COMPOSITE_SCORE = True
        if USE_COMPOSITE_SCORE:
            nc_safe = np.maximum(normal_consistencies, 0.01)
            composite_scores = (medians * (1.0 + iqrs)) / nc_safe
            order = np.argsort(composite_scores)
        else:
            # Original: median primary, IQR tie-break.
            order = np.lexsort((iqrs, medians))
        
        best_idx = int(order[0])
        best_nc = float(normal_consistencies[best_idx])

        n_top_mask = min(10, len(order))
        top_cols = order[:n_top_mask].astype(int)
        frac_in_tail = np.mean(
            burn_in_distances[:, top_cols] > global_peak_threshold, axis=1
        )
        included_mask = frac_in_tail < 0.9
        if not np.any(included_mask):
            print(
                "  Warning: no points passed top-10 @ 90% tail mask; using all points."
            )
            included_mask = np.ones(n_pts_burn, dtype=bool)
        n_included = int(np.sum(included_mask))
        print(
            f"  ICP / debug mask: top-{n_top_mask} composite-ranked tries; "
            f"exclude source point if NN > Otsu thr in ≥90% of those tries. "
            f"Included {n_included}/{n_pts_burn} ({100.0 * n_included / n_pts_burn:.1f}%)."
        )
        best_median, best_iqr = median_iqr_included(
            burn_in_distances[:, best_idx], included_mask
        )

        # Same subset as median_nn: nc on burn-in / ICP-included points only (not diluted by tail).
        _, nc_on_included, _ = validate_alignment(
            source_pcd_full,
            target_pcd_full,
            candidate_transforms[best_idx],
            {},
            11,
            global_peak_threshold=global_peak_threshold,
            include_mask=included_mask,
            quiet=True,
        )
        print(
            f"  Normal consistency on burn-in included points: {nc_on_included:.4f} "
            f"(all {n_pts_burn} source pts used for candidate ranking nc: {best_nc:.4f})"
        )

        top_k = min(5, len(order))
        if USE_COMPOSITE_SCORE:
            print(
                "Best candidates (composite: median × (1+IQR) / nc; lower is better): "
                + ", ".join(
                    f"try {int(order[j]) + 1}: med={medians[int(order[j])]:.6f} "
                    f"IQR={iqrs[int(order[j])]:.6f} nc={normal_consistencies[int(order[j])]:.4f} "
                    f"score={composite_scores[int(order[j])]:.6f}"
                    for j in range(top_k)
                )
            )
        else:
            print(
                "Best candidates (lowest median NN on included points; IQR tie-break): "
                + ", ".join(
                    f"try {int(order[j]) + 1}: med={medians[int(order[j])]:.6f} "
                    f"(IQR={iqrs[int(order[j])]:.6f})"
                    for j in range(top_k)
                )
            )
        ransac_result.transformation = candidate_transforms[best_idx].copy()
        print(
            f"Selected candidate try {best_idx + 1}/{n_ransac_attempts} "
            f"(median_nn={best_median:.6f}, IQR={best_iqr:.6f}, "
            f"nc={nc_on_included:.4f} on {n_included} included points)."
        )

        # Quality in [0,1]: higher is better (for metadata / UI); median is primary diagnostic.
        quality_score = 1.0 / (1.0 + best_median)

        chosen_dists = burn_in_distances[:, best_idx].copy()
        save_nn_distance_histogram(
            chosen_dists,
            global_peak_threshold,
            "RANSAC_chosen_NN_distances",
            debug_dir,
            subtitle_extra=(
                f"Chosen try; median={best_median:.6f} IQR={best_iqr:.6f} nc={nc_on_included:.4f}; "
                f"{n_included} included pts"
            ),
        )

        # RANSAC composite score (on included points only).
        ransac_nc = nc_on_included
        ransac_median = best_median
        ransac_score = ransac_nc / (1.0 + ransac_median)

        # Plot intermediate RANSAC result (red = included for scoring, green = excluded tail mask)
        source_ransac = copy.deepcopy(source_pcd_full)
        source_ransac.transform(ransac_result.transformation)
        self.plot_debug_step_ransac_exclusions(
            np.asarray(source_ransac.points),
            np.asarray(target_pcd_full.points),
            included_mask,
            "03_RANSAC_Result",
            debug_dir,
        )

        # Single full-res point-to-plane ICP from RANSAC init.
        icp_correspondence_dist = float(fpfh_radius_base) * 0.5
        source_full_icp = filter_source_pcd_for_icp(
            source_pcd_full,
            target_pcd_full,
            ransac_result.transformation,
            global_peak_threshold,
            "full",
            source_keep_mask=included_mask,
        )
        target_icp = copy.deepcopy(target_pcd_full)
        if not target_icp.has_normals():
            target_icp.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius_base, max_nn=normal_max_nn
                )
            )
        if not source_full_icp.has_normals():
            source_full_icp.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius_base, max_nn=normal_max_nn
                )
            )
        icp_result = o3d.pipelines.registration.registration_icp(
            source_full_icp,
            target_icp,
            max_correspondence_distance=icp_correspondence_dist,
            init=ransac_result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=120,
                relative_fitness=1e-8,
                relative_rmse=1e-8,
            ),
        )

        # Always plot the ICP result for diagnostic purposes (step 04).
        source_icp_vis = copy.deepcopy(source_full_icp)
        source_icp_vis.transform(icp_result.transformation)
        self.plot_debug_step(
            np.asarray(source_icp_vis.points),
            np.asarray(target_icp.points),
            "04_ICP_Result_full_point_to_plane",
            debug_dir,
        )

        # --- Post-ICP score on the same included-point subset ---
        _, icp_nc, _ = validate_alignment(
            source_pcd_full,
            target_pcd_full,
            icp_result.transformation,
            {},
            11,
            global_peak_threshold=global_peak_threshold,
            include_mask=included_mask,
            quiet=True,
        )
        # Median NN for ICP transform (included points only).
        _src_icp_tmp = copy.deepcopy(source_pcd_full)
        _src_icp_tmp.transform(icp_result.transformation)
        _icp_pts = np.asarray(_src_icp_tmp.points)
        _tgt_tree = o3d.geometry.KDTreeFlann(target_pcd_full)
        _icp_inc_idx = np.nonzero(np.asarray(included_mask, dtype=bool))[0]
        _icp_nn_dists = np.empty(len(_icp_inc_idx), dtype=np.float64)
        for _j, _si in enumerate(_icp_inc_idx):
            _, _, _d2 = _tgt_tree.search_knn_vector_3d(_icp_pts[_si], 1)
            _icp_nn_dists[_j] = np.sqrt(_d2[0])
        icp_median = float(np.median(_icp_nn_dists))
        icp_score = icp_nc / (1.0 + icp_median)

        # --- Choose better transform ---
        if ransac_only:
            # Upload-fit initializer: keep RANSAC pose; mesh-directed ICP runs afterward.
            chosen_transform = ransac_result.transformation
            chosen_nc = ransac_nc
            chosen_median = ransac_median
            chosen_score = ransac_score
            chosen_source = "RANSAC (upload-fit initializer, Poisson ICP skipped)"
        elif icp_score >= ransac_score:
            chosen_transform = icp_result.transformation
            chosen_nc = icp_nc
            chosen_median = icp_median
            chosen_score = icp_score
            chosen_source = "ICP"
        else:
            chosen_transform = ransac_result.transformation
            chosen_nc = ransac_nc
            chosen_median = ransac_median
            chosen_score = ransac_score
            chosen_source = "RANSAC (ICP discarded)"
            print(
                f"WARNING: ICP worsened alignment — falling back to RANSAC. "
                f"RANSAC score={ransac_score:.4f} (nc={ransac_nc:.4f}, med={ransac_median:.6f}), "
                f"ICP score={icp_score:.4f} (nc={icp_nc:.4f}, med={icp_median:.6f})"
            )
        print(
            f"Chosen transform: {chosen_source} "
            f"(score={chosen_score:.4f}, nc={chosen_nc:.4f}, median_nn={chosen_median:.6f})"
        )

        # Full cloud overlay using the chosen transform (step 05).
        source_transformed = copy.deepcopy(source_pcd_full)
        source_transformed.transform(chosen_transform)
        self.plot_debug_step(
            np.asarray(source_transformed.points),
            np.asarray(target_pcd_full.points),
            "05_After_RANSAC_and_ICP_Full",
            debug_dir,
        )

        # --- Metadata: uses the chosen (better) transform's score ---
        if update_alignment_metadata:
            alignment_score_threshold = 0.85
            alignment_failed = bool(chosen_score < alignment_score_threshold)
            try:
                with open(target_metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["alignment_error"] = alignment_failed
                metadata["alignment_score"] = float(chosen_score)
                metadata["alignment_median_nn"] = float(chosen_median)
                tmp_path = target_metadata_path + ".tmp"
                try:
                    with open(tmp_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                    os.replace(tmp_path, target_metadata_path)
                except Exception:
                    if os.path.isfile(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    raise
                print(
                    f"Metadata: alignment_error={alignment_failed} "
                    f"(composite_score={chosen_score:.4f} < {alignment_score_threshold}), "
                    f"alignment_score={chosen_score:.4f} "
                    f"(nc={chosen_nc:.4f}, median_nn={chosen_median:.6f})"
                )
            except Exception as e:
                print(f"Failed to update metadata: {str(e)}")

        # Adjust transformation matrix for denormalization - invert so that we can
        # align the target to the source instead of the source to the target.
        # NOTE: this intentionally returns the rotation + RMS-descaled residual
        # translation only (no centroid terms). Voxel-volume callers reconstruct
        # the full placement themselves by rotating in place about their own
        # centroid and then explicitly snapping source_centroid/target_centroid
        # together when embedding into the reference canvas; baking the centroid
        # offset into this matrix would double-count it there. Callers that need
        # the full raw-to-raw transform (e.g. preserved-mesh point clouds) should
        # compose it themselves via compose_source_to_target_transform using the
        # source_centroid/target_centroid/chosen_transform/normalization_scale_factor
        # also returned below.
        transformation_matrix = np.linalg.inv(chosen_transform)
        transformation_matrix = transformation_matrix.copy()
        transformation_matrix[:3, 3] *= scale_factor

        # Create homogeneous coordinates for target points 4D (x,y,z,1)
        target_pcd_full.scale(scale_factor, center)
        target_landmarks_homogeneous = np.hstack([target_pcd_full.points, np.ones((len(target_pcd_full.points), 1))])
        
        # Apply transformation to target points so they align to the source
        target_landmarks_transformed = (transformation_matrix @ target_landmarks_homogeneous.T).T[:, :3]
        
        # Get centered source landmarks and add back the source centroid
        source_pcd_full.scale(scale_factor, center)
        source_landmarks_in_source_space = np.asarray(source_pcd_full.points)  # Convert to numpy array
        
        
        # Plot final result: the adjusted source landmarks and the transformed target landmarks
        self.plot_debug_step(source_landmarks_in_source_space, 
                           target_landmarks_transformed, 
                           "06_Final_Result", debug_dir)

        # Compose PCA pre-alignment into the final matrix only when it was actually applied;
        # skipping this when do_pca=False keeps the matrix correct (PCA was never applied to the clouds).
        if do_pca:
            transformation_matrix = transformation_matrix @ pca_transformation
            print("Final transformation matrix (including PCA pre-alignment):", transformation_matrix)
        else:
            print("Final transformation matrix:", transformation_matrix)
       

        return {
            'transformation_matrix': transformation_matrix,
            'source_centroid': source_centroid,
            'target_centroid': target_centroid,
            'scale': scale_match,
            'chosen_transform': chosen_transform,
            'normalization_scale_factor': scale_factor,
        }

    def align_landmarks_to_mesh(
        self,
        source_vertices,
        source_faces,
        target_vertices,
        target_faces,
        target_metadata_path,
        scaling_mode=False,
        outer_surface=False,
    ):
        """Align reference mesh to subject mesh (standard scan-to-reference rigid registration)."""
        return self._align_landmarks_to_mesh_impl(
            source_vertices,
            source_faces,
            target_vertices,
            target_faces,
            target_metadata_path,
            scaling_mode=scaling_mode,
            outer_surface=outer_surface,
            source_landmarks=None,
            update_alignment_metadata=True,
            debug_subdir='debug_alignment',
        )

    def align_pointcloud_to_mesh(
        self,
        source_points,
        target_vertices,
        target_faces,
        target_metadata_path,
        scaling_mode=False,
        outer_surface=False,
        debug_subdir='debug_landmark_upload_fit',
        ransac_only=False,
        target_poisson_points=None,
    ):
        """
        Align an external point cloud (e.g. uploaded landmarks) to a target mesh.

        The source cloud is treated as the moving set; the target mesh is sampled with
        a fixed Poisson budget. Scan alignment metadata is not modified.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        return self._align_landmarks_to_mesh_impl(
            None,
            None,
            target_vertices,
            target_faces,
            target_metadata_path,
            scaling_mode=scaling_mode,
            outer_surface=outer_surface,
            source_landmarks=source_points,
            update_alignment_metadata=False,
            debug_subdir=debug_subdir,
            ransac_only=ransac_only,
            target_poisson_points=target_poisson_points,
        )


class LandmarkUploadAlignment(ALPACA):
    """
    Fit externally uploaded landmarks onto the current scan mesh.

    Upload-fit pipeline (mesh distance is the objective, not Poisson-cloud ICP):
      1) optional RMS size match about centroids
      2) RANSAC-only initializer on a dense Poisson target cloud (skip Poisson ICP)
      3) mesh-directed similarity ICP (landmarks <-> closest mesh points)
      4) optional surface snap
    """

    MIN_LANDMARKS = 4
    # Denser Poisson target for upload RANSAC initializer (748 landmarks -> 7480 samples).
    UPLOAD_POISSON_PER_LANDMARK = 10
    UPLOAD_POISSON_MIN = 5000
    UPLOAD_POISSON_MAX = 20000
    # Main pose refinement: each landmark gets a mesh correspondence every iteration.
    MESH_SIMILARITY_ICP_ITERS = 50
    # Manual-init refine: short ICP from the user pose (no RANSAC / flip seeds).
    MESH_REFINE_ICP_ITERS = 20
    # Stop when median mesh distance plateaus (must be improving, not drifting).
    MESH_SIMILARITY_ICP_TOL = 5e-5
    # Robust inlier cutoff for surface landmarks: median + k * MAD of outer-shell distance.
    ROBUST_INLIER_MAD_SCALE = 3.0
    # Sparse upload sets are more vulnerable to plausible 180-degree ICP basins.
    # Keep the extra multi-start work off dense uploads, where the current path is stable.
    UPLOAD_FLIP_TEST_MAX_LANDMARKS = 120

    @staticmethod
    def _cloud_centroid_and_rms(points):
        """Centroid and root-mean-square radius about that centroid."""
        points = np.asarray(points, dtype=np.float64)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        return centroid, max(rms, 1e-15)

    @staticmethod
    def _scale_about_centroid(points, centroid, scale):
        """Apply uniform scale around a fixed center."""
        points = np.asarray(points, dtype=np.float64)
        centroid = np.asarray(centroid, dtype=np.float64)
        return centroid + float(scale) * (points - centroid)

    @staticmethod
    def _median_mesh_distance(points, vertices, faces, calculator):
        """Robust mesh residual used for scale optimization."""
        distances = calculator(points, vertices, faces)
        return float(np.median(distances))

    @staticmethod
    def _umeyama_similarity_matrix(source_pts, target_pts, allow_scale=True):
        """
        Closed-form similarity transform (Umeyama) mapping source -> target.

        Returns a 4x4 matrix compatible with apply_rigid_transform (scale is in the 3x3 block).
        """
        source_pts = np.asarray(source_pts, dtype=np.float64)
        target_pts = np.asarray(target_pts, dtype=np.float64)
        if source_pts.shape != target_pts.shape or source_pts.ndim != 2 or source_pts.shape[1] != 3:
            raise ValueError("source_pts and target_pts must both be Nx3 with the same N")

        n_pts = len(source_pts)
        mu_src = np.mean(source_pts, axis=0)
        mu_tgt = np.mean(target_pts, axis=0)
        src_centered = source_pts - mu_src
        tgt_centered = target_pts - mu_tgt

        var_src = float(np.sum(src_centered * src_centered) / n_pts)
        cov = (tgt_centered.T @ src_centered) / n_pts
        u_mat, singular_vals, vt_mat = np.linalg.svd(cov)

        reflection_fix = np.eye(3, dtype=np.float64)
        if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0.0:
            reflection_fix[2, 2] = -1.0

        rot = u_mat @ reflection_fix @ vt_mat
        if allow_scale and var_src > 1e-15:
            scale = float(np.sum(singular_vals * np.diag(reflection_fix)) / var_src)
        else:
            scale = 1.0

        trans = mu_tgt - scale * (rot @ mu_src)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = scale * rot
        matrix[:3, 3] = trans
        return matrix

    @staticmethod
    def _rotation_180_about_axis(axis):
        """Return a proper 180-degree rotation matrix about a unit 3D axis."""
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return None
        axis = axis / norm
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    def _landmark_upload_180_degree_seeds(self, points):
        """
        Build upload-only ICP seeds: original plus 180-degree rotations about
        the initialized landmark cloud's principal axes, all around its centroid.
        """
        points = np.asarray(points, dtype=np.float64)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        if len(points) < self.MIN_LANDMARKS or np.linalg.norm(centered) < 1e-12:
            return [("original", points.copy())]

        try:
            covariance = np.cov(centered, rowvar=False)
            _, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return [("original", points.copy())]

        # eigh returns ascending variance; try the strongest anatomical axes first.
        axes = eigenvectors[:, ::-1].T
        seeds = [("original", points.copy())]
        for axis_index, axis in enumerate(axes, start=1):
            rotation = self._rotation_180_about_axis(axis)
            if rotation is None:
                continue
            rotated = centroid + centered @ rotation.T

            # Avoid spending ICP iterations on numerically duplicate seeds.
            is_duplicate = False
            for _, existing in seeds:
                rms_delta = float(np.sqrt(np.mean(np.sum((rotated - existing) ** 2, axis=1))))
                if rms_delta < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seeds.append((f"pca_axis_{axis_index}_180", rotated))

        return seeds

    def _score_landmark_upload_icp_result(self, points, full_vertices, full_faces, voxel_size):
        """Tail-aware full-mesh residual for choosing between upload ICP starts."""
        distances = self.calculate_landmark_distances(points, full_vertices, full_faces)
        median_dist = float(np.median(distances))
        p90_dist = float(np.percentile(distances, 90))
        p95_dist = float(np.percentile(distances, 95))
        max_dist = float(np.max(distances))
        tail_threshold = max(float(voxel_size) * 3.0, median_dist * 3.0)
        tail_fraction = float(np.mean(distances > tail_threshold))

        # Median keeps normal cases stable; p90/p95 and the explicit tail fraction
        # reject sparse wrong-flip fits that align a subset while leaving a long tail.
        score = (
            median_dist
            + 0.25 * p90_dist
            + 0.10 * p95_dist
            + tail_fraction * max(float(voxel_size), median_dist)
        )
        return {
            "score": float(score),
            "median": median_dist,
            "p90": p90_dist,
            "p95": p95_dist,
            "max": max_dist,
            "tail_fraction": tail_fraction,
            "tail_threshold": tail_threshold,
        }

    def _coarse_rms_size_match(self, landmark_points, target_vertices):
        """
        Stage 1 (scaling only): match landmark cloud extent to mesh extent about centroids.

        Returns scaled landmarks used as RANSAC input. Original landmark indices are preserved;
        the rigid stage estimates pose for this size-matched cloud.
        """
        landmark_centroid, landmark_rms = self._cloud_centroid_and_rms(landmark_points)
        mesh_centroid, mesh_rms = self._cloud_centroid_and_rms(target_vertices)
        coarse_scale = mesh_rms / landmark_rms
        scaled_landmarks = self._scale_about_centroid(
            landmark_points, landmark_centroid, coarse_scale
        )
        print(
            "Landmark upload fit stage 1: coarse RMS size match about centroids "
            f"(landmark_rms={landmark_rms:.6f}, mesh_rms={mesh_rms:.6f}, "
            f"coarse_scale={coarse_scale:.6f})"
        )
        return scaled_landmarks

    def _upload_poisson_budget(self, n_landmarks):
        """Target Poisson sample count for the upload RANSAC initializer."""
        return int(
            np.clip(
                n_landmarks * self.UPLOAD_POISSON_PER_LANDMARK,
                self.UPLOAD_POISSON_MIN,
                self.UPLOAD_POISSON_MAX,
            )
        )

    def _ransac_initialize_pose(self, landmark_points, target_vertices, target_faces, metadata_path):
        """
        Stage 2: RANSAC-only pose initializer on a dense Poisson target cloud.

        Poisson ICP is skipped because it optimizes against a sparse sample and can rotate
        a good RANSAC solution away from the true pose (especially for same-subject uploads).
        """
        poisson_budget = self._upload_poisson_budget(len(landmark_points))
        alignment_result = self.align_pointcloud_to_mesh(
            landmark_points,
            target_vertices,
            target_faces,
            metadata_path,
            scaling_mode=False,
            ransac_only=True,
            target_poisson_points=poisson_budget,
        )
        source_to_target = self.compose_source_to_target_transform(
            alignment_result['chosen_transform'],
            alignment_result['source_centroid'],
            alignment_result['target_centroid'],
            alignment_result['normalization_scale_factor'],
        )
        initialized_landmarks = self.apply_rigid_transform(landmark_points, source_to_target)
        print(
            "Landmark upload fit stage 2: RANSAC initializer complete "
            f"({len(initialized_landmarks)} landmarks, poisson_budget={poisson_budget})"
        )
        return initialized_landmarks

    def _classify_landmarks_for_registration(
        self, current, full_vertices, full_faces, outer_vertices, outer_faces, voxel_size
    ):
        """
        Split landmarks for robust registration.

        - interior: closest full-mesh point is clearly nearer than the outer shell
        - surface inliers: non-interior landmarks within a robust outer-shell distance band
        - outliers: surface landmarks beyond the band (excluded from Umeyama)
        """
        current = np.asarray(current, dtype=np.float64)
        full_closest = self.calculate_landmarks_closest_points(
            current, full_vertices, full_faces
        )
        outer_closest = self.calculate_landmarks_closest_points(
            current, outer_vertices, outer_faces
        )

        full_dist = np.linalg.norm(current - full_closest, axis=1)
        outer_dist = np.linalg.norm(current - outer_closest, axis=1)

        interior_margin = max(float(voxel_size) * 0.5, 0.02)
        interior_mask = full_dist + interior_margin < outer_dist
        surface_mask = ~interior_mask

        surface_outer_dist = outer_dist[surface_mask]
        if surface_outer_dist.size == 0:
            inlier_mask = np.ones(len(current), dtype=bool)
        else:
            med = float(np.median(surface_outer_dist))
            mad = float(np.median(np.abs(surface_outer_dist - med)))
            if mad < 1e-12:
                mad = float(np.std(surface_outer_dist)) or 1e-6
            cutoff = med + self.ROBUST_INLIER_MAD_SCALE * 1.4826 * mad
            inlier_mask = surface_mask & (outer_dist <= cutoff)

            if int(np.sum(inlier_mask)) < self.MIN_LANDMARKS:
                # Relax: keep the closest surface landmarks to the outer shell.
                surface_idx = np.flatnonzero(surface_mask)
                ranked = surface_idx[np.argsort(outer_dist[surface_idx])]
                keep_n = max(self.MIN_LANDMARKS, len(ranked) // 2)
                inlier_mask = np.zeros(len(current), dtype=bool)
                inlier_mask[ranked[:keep_n]] = True

        return inlier_mask, interior_mask, outer_closest, full_dist, outer_dist

    def _align_with_mesh_similarity_icp(
        self,
        points,
        full_vertices,
        full_faces,
        outer_vertices,
        outer_faces,
        allow_scale=True,
        voxel_size=0.05,
        max_iters=None,
        tol=None,
    ):
        """
        Stage 3: robust mesh-directed similarity ICP.

        Global pose is estimated from surface inliers aligned to the outer shell.
        Interior landmarks are detected via the full mesh but excluded from Umeyama so
        they do not bias scale or rotation. Outliers are trimmed with a MAD rule.
        """
        if max_iters is None:
            max_iters = self.MESH_SIMILARITY_ICP_ITERS
        if tol is None:
            tol = self.MESH_SIMILARITY_ICP_TOL

        current = np.asarray(points, dtype=np.float64).copy()
        eval_distances = self.calculate_landmark_distances(current, full_vertices, full_faces)
        previous_median = float(np.median(eval_distances))
        print(
            "Landmark upload fit stage 3: robust mesh ICP starting "
            f"(full_mesh_median={previous_median:.6f}, allow_scale={allow_scale})"
        )

        for iteration in range(max_iters):
            inlier_mask, interior_mask, outer_closest, full_dist, outer_dist = (
                self._classify_landmarks_for_registration(
                    current, full_vertices, full_faces, outer_vertices, outer_faces, voxel_size
                )
            )
            n_inliers = int(np.sum(inlier_mask))
            n_interior = int(np.sum(interior_mask))
            n_outliers = int(len(current) - n_inliers - n_interior)

            similarity = self._umeyama_similarity_matrix(
                current[inlier_mask],
                outer_closest[inlier_mask],
                allow_scale=allow_scale,
            )
            current = self.apply_rigid_transform(current, similarity)

            eval_distances = self.calculate_landmark_distances(current, full_vertices, full_faces)
            median_dist = float(np.median(eval_distances))
            inlier_median = float(np.median(eval_distances[inlier_mask])) if n_inliers else median_dist
            improvement = previous_median - median_dist

            print(
                f"Landmark upload fit stage 3: mesh ICP iter {iteration + 1}/{max_iters} "
                f"(full_mesh_median={median_dist:.6f}, inlier_median={inlier_median:.6f}, "
                f"inliers={n_inliers}, interior={n_interior}, outliers={n_outliers})"
            )

            # Only stop on a plateau when the fit is not getting worse.
            if abs(improvement) < tol and improvement >= 0.0:
                print(
                    f"Landmark upload fit stage 3: converged after {iteration + 1} iterations "
                    f"(delta={improvement:.2e})"
                )
                break
            previous_median = median_dist

        final_distances = self.calculate_landmark_distances(current, full_vertices, full_faces)
        print(
            "Landmark upload fit stage 3: mesh ICP finished "
            f"(median={float(np.median(final_distances)):.6f}, "
            f"p95={float(np.percentile(final_distances, 95)):.6f}, "
            f"max={float(np.max(final_distances)):.6f})"
        )
        return current

    def fit_landmarks_to_mesh(
        self,
        landmark_points,
        target_vertices,
        target_faces,
        metadata_path,
        scaling_mode=False,
        snap_to_mesh=False,
        outer_shell_vertices=None,
        outer_shell_faces=None,
        voxel_size=None,
    ):
        """
        Fit uploaded landmarks into mesh space while preserving landmark order.

        Returns:
            np.ndarray: fitted landmark positions (N, 3)
        """
        landmark_points = np.asarray(landmark_points, dtype=np.float64)
        if landmark_points.ndim != 2 or landmark_points.shape[1] != 3:
            raise ValueError("landmark_points must be an Nx3 array")
        if len(landmark_points) < self.MIN_LANDMARKS:
            raise ValueError(
                f"Need at least {self.MIN_LANDMARKS} landmarks for rigid fitting"
            )

        # Full mesh: evaluation + interior landmark detection.
        full_vertices = np.asarray(target_vertices, dtype=np.float64)
        full_faces = np.asarray(target_faces)

        # Outer shell: global pose (RANSAC + surface inliers). Falls back to full mesh.
        if outer_shell_vertices is not None and outer_shell_faces is not None:
            outer_vertices = np.asarray(outer_shell_vertices, dtype=np.float64)
            outer_faces = np.asarray(outer_shell_faces)
        else:
            outer_vertices = full_vertices
            outer_faces = full_faces

        if voxel_size is None:
            voxel_size = 0.05

        # Stage 1: optional coarse size match so RANSAC starts near the right scale.
        align_input = landmark_points
        if scaling_mode:
            align_input = self._coarse_rms_size_match(landmark_points, outer_vertices)

        # Stage 2: RANSAC initializer on the outer shell (dense Poisson, no Poisson ICP).
        fitted_landmarks = self._ransac_initialize_pose(
            align_input,
            outer_vertices,
            outer_faces,
            metadata_path,
        )

        # Stage 3: robust mesh ICP — surface inliers -> outer shell, interior excluded.
        # Sparse landmark uploads can have several plausible 180-degree basins, so only
        # for those uploads we try PCA-axis flips and choose by post-ICP mesh residual.
        if len(fitted_landmarks) <= self.UPLOAD_FLIP_TEST_MAX_LANDMARKS:
            seed_points = self._landmark_upload_180_degree_seeds(fitted_landmarks)
            print(
                "Landmark upload fit stage 3: sparse multi-start ICP "
                f"({len(seed_points)} seeds, {len(fitted_landmarks)} landmarks)"
            )

            best_name = None
            best_landmarks = None
            best_metrics = None
            for seed_name, seed in seed_points:
                print(f"Landmark upload fit stage 3: testing ICP seed '{seed_name}'")
                candidate_landmarks = self._align_with_mesh_similarity_icp(
                    seed,
                    full_vertices,
                    full_faces,
                    outer_vertices,
                    outer_faces,
                    allow_scale=scaling_mode,
                    voxel_size=voxel_size,
                )
                metrics = self._score_landmark_upload_icp_result(
                    candidate_landmarks,
                    full_vertices,
                    full_faces,
                    voxel_size,
                )
                print(
                    "Landmark upload fit stage 3: seed "
                    f"'{seed_name}' score={metrics['score']:.6f}, "
                    f"median={metrics['median']:.6f}, p90={metrics['p90']:.6f}, "
                    f"p95={metrics['p95']:.6f}, tail={metrics['tail_fraction']:.3f} "
                    f"(thr={metrics['tail_threshold']:.6f}), max={metrics['max']:.6f}"
                )

                if best_metrics is None or metrics["score"] < best_metrics["score"]:
                    best_name = seed_name
                    best_landmarks = candidate_landmarks
                    best_metrics = metrics

            if best_landmarks is None or best_metrics is None:
                raise RuntimeError("Sparse landmark upload ICP did not produce any candidates")

            fitted_landmarks = best_landmarks
            print(
                "Landmark upload fit stage 3: selected ICP seed "
                f"'{best_name}' (score={best_metrics['score']:.6f}, "
                f"median={best_metrics['median']:.6f}, p95={best_metrics['p95']:.6f})"
            )
        else:
            fitted_landmarks = self._align_with_mesh_similarity_icp(
                fitted_landmarks,
                full_vertices,
                full_faces,
                outer_vertices,
                outer_faces,
                allow_scale=scaling_mode,
                voxel_size=voxel_size,
            )

        # Stage 4: optional projection onto the mesh surface.
        if snap_to_mesh:
            fitted_landmarks = self.calculate_landmarks_closest_points(
                fitted_landmarks,
                full_vertices,
                full_faces,
            )
            print(
                f"Landmark upload fit stage 4: surface snap applied to "
                f"{len(fitted_landmarks)} landmarks"
            )
        else:
            median_dist = self._median_mesh_distance(
                fitted_landmarks, full_vertices, full_faces, self.calculate_landmark_distances
            )
            print(
                f"Landmark upload fit complete (no surface snap): "
                f"{len(fitted_landmarks)} landmarks, median_mesh_dist={median_dist:.6f}"
            )

        return fitted_landmarks

    def refine_landmarks_from_pose(
        self,
        landmark_points,
        target_vertices,
        target_faces,
        scaling_mode=False,
        outer_shell_vertices=None,
        outer_shell_faces=None,
        voxel_size=None,
    ):
        """
        Lightweight mesh ICP starting from the caller's pose.

        Does not run RANSAC, 180-degree multi-start, coarse RMS, or surface snap.
        Preserves landmark order. Intended for interactive manual initialization.
        """
        landmark_points = np.asarray(landmark_points, dtype=np.float64)
        if landmark_points.ndim != 2 or landmark_points.shape[1] != 3:
            raise ValueError("landmark_points must be an Nx3 array")
        if len(landmark_points) < self.MIN_LANDMARKS:
            raise ValueError(
                f"Need at least {self.MIN_LANDMARKS} landmarks for rigid fitting"
            )

        full_vertices = np.asarray(target_vertices, dtype=np.float64)
        full_faces = np.asarray(target_faces)

        if outer_shell_vertices is not None and outer_shell_faces is not None:
            outer_vertices = np.asarray(outer_shell_vertices, dtype=np.float64)
            outer_faces = np.asarray(outer_shell_faces)
        else:
            outer_vertices = full_vertices
            outer_faces = full_faces

        if voxel_size is None:
            voxel_size = 0.05

        print(
            "Landmark upload refine: mesh ICP from current pose "
            f"(n={len(landmark_points)}, allow_scale={bool(scaling_mode)}, "
            f"max_iters={self.MESH_REFINE_ICP_ITERS})"
        )
        refined = self._align_with_mesh_similarity_icp(
            landmark_points,
            full_vertices,
            full_faces,
            outer_vertices,
            outer_faces,
            allow_scale=bool(scaling_mode),
            voxel_size=voxel_size,
            max_iters=self.MESH_REFINE_ICP_ITERS,
        )
        median_dist = self._median_mesh_distance(
            refined, full_vertices, full_faces, self.calculate_landmark_distances
        )
        print(
            f"Landmark upload refine complete: {len(refined)} landmarks, "
            f"median_mesh_dist={median_dist:.6f}"
        )
        return refined