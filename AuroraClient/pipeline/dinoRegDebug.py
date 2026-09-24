"""
Debug artefacts for DINO-Reg rigid alignment.

Writes paper-style PCA semantic colour maps and match forensics under
``extracted/{scan}/dinoreg-rigid-debug/``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

DEBUG_SUBDIR_NAME = "dinoreg-rigid-debug"


def resolve_dino_reg_debug_dir(project_directory: str, scan_name: str) -> str:
    return os.path.join(project_directory, "extracted", scan_name, DEBUG_SUBDIR_NAME)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def pca_rgb_volume(feat_hwzc, mask_hwz):
    """
    Map the first three joint-PCA channels to RGB on the patch grid (paper-style).

    Foreground tokens are percentile-stretched per channel; background is black.
    """
    feat = np.asarray(feat_hwzc, dtype=np.float32)
    mask = np.asarray(mask_hwz, dtype=bool)
    if feat.shape[-1] < 3:
        pad = np.zeros(feat.shape[:-1] + (3 - feat.shape[-1],), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=-1)
    rgb = feat[..., :3].copy()
    if not mask.any():
        return np.zeros(mask.shape + (3,), dtype=np.float32)
    fg = rgb[mask]
    lo = np.percentile(fg, 2.0, axis=0)
    hi = np.percentile(fg, 98.0, axis=0)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[~mask] = 0.0
    return rgb.astype(np.float32)


def _max_projection_rgb(rgb_hwz3, mask_hwz):
    """Max-intensity projection along Z (numpy z axis)."""
    mask = np.asarray(mask_hwz, dtype=bool)
    rgb = np.asarray(rgb_hwz3, dtype=np.float32)
    out = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
    for c in range(3):
        plane = np.where(mask, rgb[..., c], 0.0)
        out[..., c] = plane.max(axis=2)
    return out


class DinoRegDebugWriter:
    """Collect images + JSON forensics for one DINO-Reg rigid attempt."""

    def __init__(self, debug_dir: str, scan_name: str = "", reference_name: str = ""):
        self.debug_dir = debug_dir
        self.scan_name = scan_name or "scan"
        self.reference_name = reference_name or "reference"
        os.makedirs(debug_dir, exist_ok=True)
        self.meta: Dict[str, Any] = {
            "scan": self.scan_name,
            "reference": self.reference_name,
            "debug_dir": debug_dir,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pyramid_levels": [],
        }
        readme = os.path.join(debug_dir, "00_README.txt")
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(
                "DINO-Reg rigid debug outputs\n"
                "----------------------------\n"
                "Pass 1 — cheap octahedral (24 proper 90° relabels, not mirrors):\n"
                "  octahedral_pass1.txt — cosine scoreboard (winner marked)\n"
                "  overlay_octahedral_pass1.png — FG mesh at the winning cube\n"
                "\n"
                "Pass 2 — high-res tri-planar PCA + Adam refine:\n"
                "  moving/fixed_full_pca_rgb_axial_mid.png — semantic PCA mid-slice\n"
                "  overlay_octahedral_refine_coarse/fine.png — Adam from the cube\n"
                "  overlay_accepted.png — pose actually returned\n"
                "\n"
                "If the cube is a weak winner or refine cosine is low:\n"
                "  k_trials_summary.txt / overlay_kXX_*.png / overlay_k_trials_grid.png\n"
                "  overlay_best_clusters_side_by_side.png / landmark_matches.png\n"
                "  overlay_cluster_refine_coarse/fine.png — Adam from cluster Kabsch\n"
                "\n"
                "See run_summary.json for numeric forensics.\n"
                "After apply: apply_warp_forensics.json and apply_mip_overlay.png.\n"
            )

    def set_run_context(self, **kwargs):
        self.meta["run"] = kwargs

    def record_pyramid_level(self, record: Dict[str, Any]):
        self.meta["pyramid_levels"].append(record)

    def set_outcome(self, success: bool, message: str = "", rigid: Optional[Dict[str, Any]] = None):
        self.meta["success"] = bool(success)
        self.meta["message"] = message
        if rigid is not None:
            self.meta["accepted_rigid"] = rigid

    def save_pca_semantic_views(
        self,
        feat_hwzc,
        mask_hwz,
        prefix: str,
        title_prefix: str = "",
        scales_mm=None,
    ):
        """Save axial mid-slice of PCA RGB (physical XY aspect)."""
        rgb = pca_rgb_volume(feat_hwzc, mask_hwz)
        mask = np.asarray(mask_hwz, dtype=bool)
        mid_z = int(mask.shape[2] // 2)
        self._save_rgb_image(
            rgb[:, :, mid_z],
            mask[:, :, mid_z],
            f"{prefix}_pca_rgb_axial_mid.png",
            f"{title_prefix} PCA RGB axial z={mid_z}",
            scales_mm=scales_mm,
        )

    def save_pyramid_match_figure(
        self,
        mov_pts,
        fix_pts,
        matches,
        pool: int,
        gate_mm: float,
        status: str,
    ):
        """XY/XZ/YZ scatter of mutual-NN pairs (moving vs fixed mm centres)."""
        if not matches:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mov_pts = np.asarray(mov_pts, dtype=np.float64)
        fix_pts = np.asarray(fix_pts, dtype=np.float64)
        mov_m = np.stack([mov_pts[mi] for _s, mi, _fi in matches], axis=0)
        fix_m = np.stack([fix_pts[fi] for _s, _mi, fi in matches], axis=0)
        scores = np.array([s for s, _mi, _fi in matches], dtype=np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        planes = [
            (0, 1, "XY"),
            (0, 2, "XZ"),
            (1, 2, "YZ"),
        ]
        for ax, (a, b, name) in zip(axes, planes):
            ax.scatter(mov_m[:, a], mov_m[:, b], c="tab:blue", s=12, alpha=0.7, label="moving")
            ax.scatter(fix_m[:, a], fix_m[:, b], c="tab:orange", s=12, alpha=0.7, label="fixed")
            for i in range(len(matches)):
                ax.plot(
                    [mov_m[i, a], fix_m[i, a]],
                    [mov_m[i, b], fix_m[i, b]],
                    color="gray",
                    alpha=0.25,
                    linewidth=0.6,
                )
            ax.set_xlabel(f"{name[0]} (mm)")
            ax.set_ylabel(f"{name[1]} (mm)")
            ax.set_title(name)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"÷{pool} ({len(matches)} pairs) — {status}",
            fontsize=11,
        )
        fig.tight_layout()
        path = os.path.join(self.debug_dir, f"pyramid_div{pool}_matches.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(scores, bins=min(20, max(5, len(scores))))
        ax2.set_xlabel("cosine similarity")
        ax2.set_ylabel("count")
        ax2.set_title(f"÷{pool} match score histogram (n={len(scores)})")
        ax2.grid(True, alpha=0.25)
        fig2.tight_layout()
        path2 = os.path.join(self.debug_dir, f"pyramid_div{pool}_match_scores.png")
        fig2.savefig(path2, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

    def save_cluster_label_views(self, label_hwz, mask_hwz, k: int, prefix: str):
        """Colour a joint-k-means label volume (same ids in moving and fixed)."""
        labels = np.asarray(label_hwz, dtype=np.int32)
        mask = np.asarray(mask_hwz, dtype=bool)
        n = max(int(k), 1)
        hues = np.linspace(0.0, 1.0, n, endpoint=False)
        palette = np.zeros((n, 3), dtype=np.float32)
        for i, h in enumerate(hues):
            s, v = 0.75, 0.95
            i6 = int(h * 6.0) % 6
            f = h * 6.0 - int(h * 6.0)
            p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
            palette[i] = [
                (v, t, p, p, q, v)[i6],
                (q, v, v, t, p, p)[i6],
                (p, p, q, v, v, t)[i6],
            ]
        rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
        keep = mask & (labels >= 0)
        if keep.any():
            rgb[keep] = palette[np.clip(labels[keep], 0, n - 1)]
        # Labels are scattered at token stride (often every 2nd Z); shape//2 can be empty.
        labeled_per_z = keep.sum(axis=(0, 1)) if keep.any() else np.zeros(labels.shape[2], dtype=np.int64)
        mid_z = int(np.argmax(labeled_per_z)) if int(labeled_per_z.max()) > 0 else int(labels.shape[2] // 2)
        self._save_rgb_image(
            rgb[:, :, mid_z],
            mask[:, :, mid_z],
            f"{prefix}_axial_mid.png",
            f"{prefix} semantic clusters axial z={mid_z}",
        )

    def save_rigid_mesh_overlay(
        self,
        mov_mask,
        fix_mask,
        mov_origin_mm,
        mov_scales_mm,
        fix_origin_mm,
        fix_scales_mm,
        R,
        t,
        filename: str,
        title: str = "",
        elev: float = -28.0,
        azim: float = 35.0,
        mov_centroids_mm=None,
        fix_centroids_mm=None,
        target_vox: int = 160,
        dpi: int = 180,
    ):
        """
        Angled 3D marching-cubes overlay: fixed (cyan) vs moving warped by R,t (coral).

        Optional homologous centroids are drawn color-matched (same index → same color);
        fixed = filled circle, moving-warped = triangle.
        """
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage import measure
        except Exception as exc:
            with open(
                os.path.join(self.debug_dir, f"{filename}.error.txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(f"import failed: {exc}\n")
            return

        def _mesh(mask, origin, scales):
            mask = np.asarray(mask, dtype=bool)
            if int(mask.sum()) < 8:
                return None, None
            # Higher-res default (~160) so feature/native masks look less blocky.
            step = max(1, int(np.ceil(max(mask.shape) / float(target_vox))))
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(np.float32),
                    level=0.5,
                    spacing=tuple(float(s) for s in scales),
                    step_size=step,
                )
            except Exception:
                return None, None
            if verts is None or len(verts) < 4 or faces is None or len(faces) < 1:
                return None, None
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            verts = np.asarray(verts, dtype=np.float64) + origin
            return verts, np.asarray(faces, dtype=np.int64)

        mov_v, mov_f = _mesh(mov_mask, mov_origin_mm, mov_scales_mm)
        fix_v, fix_f = _mesh(fix_mask, fix_origin_mm, fix_scales_mm)
        if fix_v is None and mov_v is None:
            return

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        if mov_v is not None:
            mov_v = (R @ mov_v.T).T + t

        mov_c = None if mov_centroids_mm is None else np.asarray(mov_centroids_mm, dtype=np.float64)
        fix_c = None if fix_centroids_mm is None else np.asarray(fix_centroids_mm, dtype=np.float64)
        if mov_c is not None and len(mov_c):
            mov_c = (R @ mov_c.T).T + t

        fig = plt.figure(figsize=(8.5, 8.5), facecolor="black")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        def _add(verts, faces, rgba):
            if verts is None or faces is None:
                return
            tris = verts[faces]
            coll = Poly3DCollection(tris, alpha=float(rgba[3]), linewidths=0.0)
            coll.set_facecolor(rgba)
            coll.set_edgecolor((0, 0, 0, 0))
            ax.add_collection3d(coll)

        _add(fix_v, fix_f, (0.25, 0.85, 0.95, 0.50))
        _add(mov_v, mov_f, (1.00, 0.45, 0.30, 0.50))

        pts = []
        if fix_v is not None:
            pts.append(fix_v)
        if mov_v is not None:
            pts.append(mov_v)
        if fix_c is not None and len(fix_c):
            pts.append(fix_c)
        if mov_c is not None and len(mov_c):
            pts.append(mov_c)

        n_lm = 0
        if fix_c is not None and mov_c is not None:
            n_lm = int(min(len(fix_c), len(mov_c)))
        if n_lm > 0:
            cmap = cm.get_cmap("tab20" if n_lm > 10 else "tab10")
            for i in range(n_lm):
                color = cmap(i % cmap.N)
                ax.scatter(
                    [fix_c[i, 0]], [fix_c[i, 1]], [fix_c[i, 2]],
                    c=[color], s=55, marker="o", depthshade=False, edgecolors="white", linewidths=0.4,
                )
                ax.scatter(
                    [mov_c[i, 0]], [mov_c[i, 1]], [mov_c[i, 2]],
                    c=[color], s=70, marker="^", depthshade=False, edgecolors="white", linewidths=0.4,
                )
                ax.plot(
                    [fix_c[i, 0], mov_c[i, 0]],
                    [fix_c[i, 1], mov_c[i, 1]],
                    [fix_c[i, 2], mov_c[i, 2]],
                    color=color, alpha=0.55, linewidth=1.2,
                )

        all_pts = np.concatenate(pts, axis=0)
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        mid = 0.5 * (lo + hi)
        span = float(np.max(hi - lo))
        half = 0.55 * max(span, 1.0)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=float(elev), azim=float(azim))
        if title:
            ax.set_title(title, color="white", fontsize=10, pad=2)
        fig.text(
            0.5,
            0.02,
            "cyan=fixed  coral=moving@R,t  ○ fixed CoM  △ moving CoM (matched colors)",
            ha="center",
            color="0.7",
            fontsize=8,
        )
        out = os.path.join(self.debug_dir, filename)
        fig.savefig(out, facecolor="black", bbox_inches="tight", pad_inches=0.05, dpi=int(dpi))
        plt.close(fig)

    def save_k_trials_grid(
        self,
        trial_overlays,
        best_k=None,
        filename: str = "overlay_k_trials_grid.png",
    ):
        """
        One big grid: each successful K as a row with 3D overlay + XY/XZ/YZ projections.

        ``trial_overlays`` items: dict with keys
        k, R, t, angle_deg, n_inliers, n_landmarks, mov_mask, fix_mask,
        mov_origin, mov_scales, fix_origin, fix_scales, mov_centroids, fix_centroids.
        """
        rows = [r for r in (trial_overlays or []) if r.get("R") is not None]
        if not rows:
            return
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage import measure
        except Exception as exc:
            with open(
                os.path.join(self.debug_dir, f"{filename}.error.txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(f"import failed: {exc}\n")
            return

        def _mesh(mask, origin, scales, target_vox=120):
            mask = np.asarray(mask, dtype=bool)
            if int(mask.sum()) < 8:
                return None, None
            step = max(1, int(np.ceil(max(mask.shape) / float(target_vox))))
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(np.float32),
                    level=0.5,
                    spacing=tuple(float(s) for s in scales),
                    step_size=step,
                )
            except Exception:
                return None, None
            if verts is None or len(verts) < 4 or faces is None or len(faces) < 1:
                return None, None
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            return np.asarray(verts, dtype=np.float64) + origin, np.asarray(faces, dtype=np.int64)

        def _mip(mask, axis):
            m = np.asarray(mask, dtype=bool)
            return m.any(axis=axis).astype(np.float32)

        n_rows = len(rows)
        fig = plt.figure(figsize=(16, 3.6 * n_rows), facecolor="#0b0b0b")
        gs = fig.add_gridspec(n_rows, 4, wspace=0.08, hspace=0.22)

        for r_i, row in enumerate(rows):
            R = np.asarray(row["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(row["t"], dtype=np.float64).reshape(3)
            mov_c = np.asarray(row["mov_centroids"], dtype=np.float64)
            fix_c = np.asarray(row["fix_centroids"], dtype=np.float64)
            mov_cw = (R @ mov_c.T).T + t if len(mov_c) else mov_c
            n_lm = int(min(len(mov_cw), len(fix_c)))
            cmap = cm.get_cmap("tab20" if n_lm > 10 else "tab10")
            colors = [cmap(i % cmap.N) for i in range(max(n_lm, 1))]

            # --- 3D panel ---
            ax3 = fig.add_subplot(gs[r_i, 0], projection="3d")
            ax3.set_facecolor("black")
            for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("none")
            ax3.grid(False)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_zticks([])
            ax3.set_axis_off()

            mov_v, mov_f = _mesh(row["mov_mask"], row["mov_origin"], row["mov_scales"])
            fix_v, fix_f = _mesh(row["fix_mask"], row["fix_origin"], row["fix_scales"])
            if mov_v is not None:
                mov_v = (R @ mov_v.T).T + t

            def _add(ax, verts, faces, rgba):
                if verts is None or faces is None:
                    return
                coll = Poly3DCollection(verts[faces], alpha=float(rgba[3]), linewidths=0.0)
                coll.set_facecolor(rgba)
                coll.set_edgecolor((0, 0, 0, 0))
                ax.add_collection3d(coll)

            _add(ax3, fix_v, fix_f, (0.25, 0.85, 0.95, 0.50))
            _add(ax3, mov_v, mov_f, (1.00, 0.45, 0.30, 0.50))
            for i in range(n_lm):
                ax3.scatter(
                    [fix_c[i, 0]], [fix_c[i, 1]], [fix_c[i, 2]],
                    c=[colors[i]], s=36, marker="o", depthshade=False, edgecolors="white", linewidths=0.3,
                )
                ax3.scatter(
                    [mov_cw[i, 0]], [mov_cw[i, 1]], [mov_cw[i, 2]],
                    c=[colors[i]], s=48, marker="^", depthshade=False, edgecolors="white", linewidths=0.3,
                )
                ax3.plot(
                    [fix_c[i, 0], mov_cw[i, 0]],
                    [fix_c[i, 1], mov_cw[i, 1]],
                    [fix_c[i, 2], mov_cw[i, 2]],
                    color=colors[i], alpha=0.5, linewidth=1.0,
                )
            pts = []
            for v in (fix_v, mov_v):
                if v is not None:
                    pts.append(v)
            if n_lm:
                pts.extend([fix_c[:n_lm], mov_cw[:n_lm]])
            if pts:
                all_pts = np.concatenate(pts, axis=0)
                mid = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
                half = 0.55 * max(float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0))), 1.0)
                ax3.set_xlim(mid[0] - half, mid[0] + half)
                ax3.set_ylim(mid[1] - half, mid[1] + half)
                ax3.set_zlim(mid[2] - half, mid[2] + half)
                ax3.set_box_aspect((1, 1, 1))
            ax3.view_init(elev=-28, azim=35)
            mark = " ★" if best_k is not None and int(row["k"]) == int(best_k) else ""
            ax3.set_title(
                f"K={row['k']}{mark}  cos={float(row.get('cosine', float('nan'))):.3f}  "
                f"{float(row.get('angle_deg', 0)):.1f}°  "
                f"in={row.get('n_inliers')}/{row.get('n_landmarks')}",
                color="white",
                fontsize=9,
                pad=1,
            )

            # --- projection panels (fixed MIP + CoMs) ---
            # Build a coarse fixed-space occupancy of warped moving for context.
            fix_mask = np.asarray(row["fix_mask"], dtype=bool)
            fix_origin = np.asarray(row["fix_origin"], dtype=np.float64).reshape(3)
            fix_scales = np.asarray(row["fix_scales"], dtype=np.float64).reshape(3)

            planes = [
                (0, 1, 2, "XY"),   # project along Z
                (0, 2, 1, "XZ"),   # along Y
                (1, 2, 0, "YZ"),   # along X
            ]
            for c_i, (a, b, proj_axis, name) in enumerate(planes):
                axp = fig.add_subplot(gs[r_i, c_i + 1])
                axp.set_facecolor("black")
                mip = _mip(fix_mask, axis=proj_axis)
                # Extent in mm for displayed axes (x=b horizontal, y=a vertical).
                dims = [0, 1, 2]
                dims.remove(proj_axis)
                d0, d1 = dims[0], dims[1]
                # mip axes follow remaining dims in ascending numpy order.
                if (d0, d1) == (a, b):
                    img = mip
                    ext = [
                        float(fix_origin[b]),
                        float(fix_origin[b] + fix_scales[b] * mip.shape[1]),
                        float(fix_origin[a]),
                        float(fix_origin[a] + fix_scales[a] * mip.shape[0]),
                    ]
                else:
                    img = mip.T
                    ext = [
                        float(fix_origin[b]),
                        float(fix_origin[b] + fix_scales[b] * mip.shape[0]),
                        float(fix_origin[a]),
                        float(fix_origin[a] + fix_scales[a] * mip.shape[1]),
                    ]
                axp.imshow(
                    img,
                    origin="lower",
                    cmap="Blues",
                    alpha=0.85,
                    extent=ext,
                    aspect="equal",
                )
                for i in range(n_lm):
                    axp.scatter(
                        [fix_c[i, b]], [fix_c[i, a]],
                        c=[colors[i]], s=42, marker="o", edgecolors="white", linewidths=0.4, zorder=3,
                    )
                    axp.scatter(
                        [mov_cw[i, b]], [mov_cw[i, a]],
                        c=[colors[i]], s=55, marker="^", edgecolors="white", linewidths=0.4, zorder=3,
                    )
                    axp.plot(
                        [fix_c[i, b], mov_cw[i, b]],
                        [fix_c[i, a], mov_cw[i, a]],
                        color=colors[i], alpha=0.55, linewidth=1.0, zorder=2,
                    )
                axp.set_title(name, color="white", fontsize=8)
                axp.tick_params(colors="0.55", labelsize=6)
                for spine in axp.spines.values():
                    spine.set_color("0.35")

        fig.suptitle(
            "Semantic-landmark Kabsch trials — cyan/coral meshes, color-matched CoMs (○ fixed, △ moving@R,t)",
            color="white",
            fontsize=11,
            y=0.995,
        )
        out = os.path.join(self.debug_dir, filename)
        fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08, dpi=150)
        plt.close(fig)

    def save_k_trials_summary(self, trials, best_k=None):
        """Write a human-readable multi-K Kabsch scoreboard next to the PNGs."""
        path = os.path.join(self.debug_dir, "k_trials_summary.txt")
        lines = [
            "Semantic-landmark Kabsch trials (cluster CoMs → weighted Kabsch)",
            "Ranked by masked feature cosine (inlier count is diagnostic only).",
            "K is the joint k-means cluster count; each trial is an independent pose.",
            "",
        ]
        for row in trials or []:
            k = row.get("k")
            status = row.get("pose_status") or row.get("landmark_status")
            mark = " <-- best" if best_k is not None and int(k) == int(best_k) else ""
            if row.get("pose_status") == "ok":
                cos = row.get("cosine", float("nan"))
                lines.append(
                    f"K={k}: Kabsch ok  cos={cos:.3f}  landmarks={row.get('n_landmarks', '?')}  "
                    f"inliers={row.get('n_inliers')}  "
                    f"med_res={row.get('median_residual_mm', float('nan')):.2f} mm  "
                    f"angle={row.get('angle_deg', float('nan')):.1f}°  "
                    f"label={row.get('label')}{mark}"
                )
            else:
                lines.append(
                    f"K={k}: {status}  landmarks={row.get('n_landmarks', 0)}{mark}"
                )
        lines.append("")
        lines.append(f"best_k={best_k}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def save_match_landmarks_side_by_side(
        self,
        mov_mask,
        fix_mask,
        mov_origin_mm,
        mov_scales_mm,
        fix_origin_mm,
        fix_scales_mm,
        R,
        t,
        mov_pts_mm,
        fix_pts_mm,
        scores=None,
        filename: str = "overlay_best_matches_side_by_side.png",
        title: str = "",
        elev: float = -28.0,
        azim: float = 35.0,
        target_vox: int = 180,
        max_landmarks: int = 48,
    ):
        """
        Side-by-side 3D: fixed FG mesh | moving FG mesh warped by R,t.

        Mutual-NN (or RANSAC-inlier) match pairs are drawn as landmarks —
        ○ fixed, △ moving@R,t — same index → same colour. Highest cosine
        scores are kept first (up to ``max_landmarks``).
        """
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage import measure
        except Exception as exc:
            with open(
                os.path.join(self.debug_dir, f"{filename}.error.txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(f"import failed: {exc}\n")
            return

        mov_pts = np.asarray(mov_pts_mm, dtype=np.float64).reshape(-1, 3)
        fix_pts = np.asarray(fix_pts_mm, dtype=np.float64).reshape(-1, 3)
        n = int(min(len(mov_pts), len(fix_pts)))
        if n < 1:
            return
        mov_pts = mov_pts[:n]
        fix_pts = fix_pts[:n]
        if scores is None:
            order = np.arange(n)
            sc = np.ones(n, dtype=np.float64)
        else:
            sc = np.asarray(scores, dtype=np.float64).reshape(-1)[:n]
            order = np.argsort(-sc)  # highest cosine first
        if n > int(max_landmarks):
            order = order[: int(max_landmarks)]
        mov_pts = mov_pts[order]
        fix_pts = fix_pts[order]
        sc = sc[order]
        n = len(mov_pts)

        def _mesh(mask, origin, scales):
            mask = np.asarray(mask, dtype=bool)
            if int(mask.sum()) < 8:
                return None, None
            step = max(1, int(np.ceil(max(mask.shape) / float(target_vox))))
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(np.float32),
                    level=0.5,
                    spacing=tuple(float(s) for s in scales),
                    step_size=step,
                )
            except Exception:
                return None, None
            if verts is None or len(verts) < 4 or faces is None or len(faces) < 1:
                return None, None
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            return np.asarray(verts, dtype=np.float64) + origin, np.asarray(faces, dtype=np.int64)

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        fix_v, fix_f = _mesh(fix_mask, fix_origin_mm, fix_scales_mm)
        mov_v, mov_f = _mesh(mov_mask, mov_origin_mm, mov_scales_mm)
        if mov_v is not None:
            mov_v = (R @ mov_v.T).T + t
        mov_w = (R @ mov_pts.T).T + t

        fig = plt.figure(figsize=(14, 7), facecolor="black")
        ax_f = fig.add_subplot(121, projection="3d")
        ax_m = fig.add_subplot(122, projection="3d")
        for ax, name in ((ax_f, "fixed + match landmarks"), (ax_m, "moving@R,t + match landmarks")):
            ax.set_facecolor("black")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("none")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            ax.set_title(name, color="white", fontsize=10, pad=2)

        def _add_mesh(ax, verts, faces, rgba):
            if verts is None or faces is None:
                return
            coll = Poly3DCollection(verts[faces], alpha=float(rgba[3]), linewidths=0.0)
            coll.set_facecolor(rgba)
            coll.set_edgecolor((0, 0, 0, 0))
            ax.add_collection3d(coll)

        _add_mesh(ax_f, fix_v, fix_f, (0.55, 0.58, 0.62, 0.22))
        _add_mesh(ax_m, mov_v, mov_f, (0.55, 0.58, 0.62, 0.22))

        cmap = cm.get_cmap("tab20" if n > 10 else "tab10")
        all_pts = []
        if fix_v is not None:
            all_pts.append(fix_v)
        if mov_v is not None:
            all_pts.append(mov_v)
        for i in range(n):
            col = cmap(i % cmap.N)
            ax_f.scatter(
                [fix_pts[i, 0]], [fix_pts[i, 1]], [fix_pts[i, 2]],
                c=[col], s=48, marker="o", depthshade=False,
                edgecolors="white", linewidths=0.4, zorder=5,
            )
            ax_m.scatter(
                [mov_w[i, 0]], [mov_w[i, 1]], [mov_w[i, 2]],
                c=[col], s=60, marker="^", depthshade=False,
                edgecolors="white", linewidths=0.4, zorder=5,
            )
            all_pts.append(fix_pts[i:i + 1])
            all_pts.append(mov_w[i:i + 1])

        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            mid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            half = 0.55 * max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), 1.0)
            for ax in (ax_f, ax_m):
                ax.set_xlim(mid[0] - half, mid[0] + half)
                ax.set_ylim(mid[1] - half, mid[1] + half)
                ax.set_zlim(mid[2] - half, mid[2] + half)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=float(elev), azim=float(azim))

        if title:
            fig.suptitle(title, color="white", fontsize=11)
        sc_lo = float(sc.min()) if len(sc) else 0.0
        sc_hi = float(sc.max()) if len(sc) else 0.0
        fig.text(
            0.5,
            0.02,
            f"{n} mutual-NN landmarks (top cosine {sc_hi:.3f}→{sc_lo:.3f})  "
            f"○ fixed  △ moving@R,t  — same colour = same match",
            ha="center",
            color="0.7",
            fontsize=8,
        )
        out = os.path.join(self.debug_dir, filename)
        fig.savefig(out, facecolor="black", bbox_inches="tight", pad_inches=0.06, dpi=160)
        plt.close(fig)

    def save_winning_cluster_side_by_side(
        self,
        mov_labels,
        fix_labels,
        mov_mask,
        fix_mask,
        mov_origin_mm,
        mov_scales_mm,
        fix_origin_mm,
        fix_scales_mm,
        R,
        t,
        k: int,
        mov_centroids_mm=None,
        fix_centroids_mm=None,
        filename: str = "overlay_best_clusters_side_by_side.png",
        title: str = "",
        elev: float = -28.0,
        azim: float = 35.0,
        target_vox: int = 180,
        # Optional native-resolution FG masks + geometry (preferred for sharp meshes).
        mov_mask_native=None,
        fix_mask_native=None,
        mov_origin_native_mm=None,
        mov_scales_native_mm=None,
        fix_origin_native_mm=None,
        fix_scales_native_mm=None,
    ):
        """
        Side-by-side 3D: fixed clusters | moving clusters warped by R,t.

        Each joint k-means cluster id is one colour on both panels so homology
        is readable without a silhouette overlay. When native masks are passed,
        feature-grid labels are nearest-neighbour upsampled onto them for
        higher-resolution marching cubes.
        """
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from scipy import ndimage
            from skimage import measure
            from skimage.transform import resize
        except Exception as exc:
            with open(
                os.path.join(self.debug_dir, f"{filename}.error.txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write(f"import failed: {exc}\n")
            return

        def _densify(labels, mask):
            labels = np.asarray(labels, dtype=np.int32)
            mask = np.asarray(mask, dtype=bool)
            out = labels.copy()
            known = labels >= 0
            if not known.any() or not mask.any():
                return out
            need = mask & ~known
            if not need.any():
                return out
            _, (i0, j0, k0) = ndimage.distance_transform_edt(
                ~known, return_indices=True
            )
            out[need] = labels[i0[need], j0[need], k0[need]]
            out[~mask] = -1
            return out

        def _upsample_labels(labels, target_mask):
            """Nearest-neighbour upsample feature labels onto a finer FG mask."""
            labels = np.asarray(labels, dtype=np.float32)
            target_mask = np.asarray(target_mask, dtype=bool)
            if labels.shape == target_mask.shape:
                out = np.rint(labels).astype(np.int32)
                out[~target_mask] = -1
                return out
            # Shift -1 → 0 temporarily so resize stays well-defined, then restore.
            shifted = labels + 1.0  # unlabeled 0, cluster ids 1..
            up = resize(
                shifted,
                target_mask.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            out = np.rint(up).astype(np.int32) - 1
            out[~target_mask] = -1
            return out

        def _mesh_cluster(label_vol, cid, origin, scales):
            m = np.asarray(label_vol, dtype=np.int32) == int(cid)
            if int(m.sum()) < 8:
                return None, None
            step = max(1, int(np.ceil(max(m.shape) / float(target_vox))))
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    m.astype(np.float32),
                    level=0.5,
                    spacing=tuple(float(s) for s in scales),
                    step_size=step,
                )
            except Exception:
                return None, None
            if verts is None or len(verts) < 4 or faces is None or len(faces) < 1:
                return None, None
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            return np.asarray(verts, dtype=np.float64) + origin, np.asarray(faces, dtype=np.int64)

        use_native = (
            mov_mask_native is not None
            and fix_mask_native is not None
            and mov_origin_native_mm is not None
            and fix_origin_native_mm is not None
            and mov_scales_native_mm is not None
            and fix_scales_native_mm is not None
        )
        if use_native:
            mov_lab = _densify(
                _upsample_labels(mov_labels, mov_mask_native), mov_mask_native
            )
            fix_lab = _densify(
                _upsample_labels(fix_labels, fix_mask_native), fix_mask_native
            )
            mov_origin_mm = mov_origin_native_mm
            fix_origin_mm = fix_origin_native_mm
            mov_scales_mm = mov_scales_native_mm
            fix_scales_mm = fix_scales_native_mm
        else:
            mov_lab = _densify(mov_labels, mov_mask)
            fix_lab = _densify(fix_labels, fix_mask)

        ids = sorted(
            set(int(x) for x in np.unique(mov_lab) if int(x) >= 0)
            | set(int(x) for x in np.unique(fix_lab) if int(x) >= 0)
        )
        if not ids:
            return

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        cmap = cm.get_cmap("tab20" if len(ids) > 10 else "tab10")
        color_of = {cid: cmap(i % cmap.N) for i, cid in enumerate(ids)}

        fig = plt.figure(figsize=(14, 7), facecolor="black")
        ax_f = fig.add_subplot(121, projection="3d")
        ax_m = fig.add_subplot(122, projection="3d")
        for ax, name in ((ax_f, "fixed clusters"), (ax_m, "moving@R,t clusters")):
            ax.set_facecolor("black")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("none")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            ax.set_title(name, color="white", fontsize=10, pad=2)

        all_pts = []

        def _add(ax, verts, faces, rgba):
            if verts is None or faces is None:
                return
            coll = Poly3DCollection(verts[faces], alpha=0.55, linewidths=0.0)
            coll.set_facecolor(rgba)
            coll.set_edgecolor((0, 0, 0, 0))
            ax.add_collection3d(coll)
            all_pts.append(verts)

        for cid in ids:
            rgba = color_of[cid]
            fv, ff = _mesh_cluster(fix_lab, cid, fix_origin_mm, fix_scales_mm)
            _add(ax_f, fv, ff, rgba)
            mv, mf = _mesh_cluster(mov_lab, cid, mov_origin_mm, mov_scales_mm)
            if mv is not None:
                mv = (R @ mv.T).T + t
            _add(ax_m, mv, mf, rgba)

        mov_c = None if mov_centroids_mm is None else np.asarray(mov_centroids_mm, dtype=np.float64)
        fix_c = None if fix_centroids_mm is None else np.asarray(fix_centroids_mm, dtype=np.float64)
        if mov_c is not None and len(mov_c):
            mov_c = (R @ mov_c.T).T + t
        n_lm = 0
        if fix_c is not None and mov_c is not None:
            n_lm = int(min(len(fix_c), len(mov_c)))
        for i in range(n_lm):
            col = cmap(i % cmap.N)
            ax_f.scatter(
                [fix_c[i, 0]], [fix_c[i, 1]], [fix_c[i, 2]],
                c=[col], s=48, marker="o", depthshade=False, edgecolors="white", linewidths=0.4,
            )
            ax_m.scatter(
                [mov_c[i, 0]], [mov_c[i, 1]], [mov_c[i, 2]],
                c=[col], s=60, marker="^", depthshade=False, edgecolors="white", linewidths=0.4,
            )
            all_pts.extend([fix_c[i:i + 1], mov_c[i:i + 1]])

        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            mid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            half = 0.55 * max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), 1.0)
            for ax in (ax_f, ax_m):
                ax.set_xlim(mid[0] - half, mid[0] + half)
                ax.set_ylim(mid[1] - half, mid[1] + half)
                ax.set_zlim(mid[2] - half, mid[2] + half)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=float(elev), azim=float(azim))

        if title:
            fig.suptitle(title, color="white", fontsize=11)
        res_note = "native-res" if use_native else "feature-grid"
        fig.text(
            0.5,
            0.02,
            f"K={k} joint clusters ({res_note}) — same id → same colour  ○ fixed CoM  △ moving CoM",
            ha="center",
            color="0.7",
            fontsize=8,
        )
        out = os.path.join(self.debug_dir, filename)
        fig.savefig(out, facecolor="black", bbox_inches="tight", pad_inches=0.06, dpi=160)
        plt.close(fig)

    def save_landmark_figure(self, mov_pts, fix_pts, scores, status: str = ""):
        """XY/XZ/YZ of homologous cluster centres of mass."""
        mov_pts = np.asarray(mov_pts, dtype=np.float64)
        fix_pts = np.asarray(fix_pts, dtype=np.float64)
        if len(mov_pts) == 0:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        planes = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
        for ax, (a, b, name) in zip(axes, planes):
            ax.scatter(mov_pts[:, a], mov_pts[:, b], c="tab:blue", s=36, label="moving CoM")
            ax.scatter(fix_pts[:, a], fix_pts[:, b], c="tab:orange", s=36, label="fixed CoM")
            for i in range(len(mov_pts)):
                ax.plot(
                    [mov_pts[i, a], fix_pts[i, a]],
                    [mov_pts[i, b], fix_pts[i, b]],
                    color="gray",
                    alpha=0.45,
                    linewidth=1.0,
                )
            ax.set_xlabel(f"{name[0]} (mm)")
            ax.set_ylabel(f"{name[1]} (mm)")
            ax.set_title(name)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Semantic-mass landmarks ({len(mov_pts)} clusters) — {status}")
        fig.tight_layout()
        path = os.path.join(self.debug_dir, "landmark_matches.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def save_pyramid_summary_chart(self):
        """Bar chart of mutual-NN counts and RANSAC status per pyramid level."""
        levels = self.meta.get("pyramid_levels") or []
        if not levels:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pools = [int(lv.get("pool", 0)) for lv in levels]
        labels = [f"÷{p}" for p in pools]
        n_match = [int(lv.get("n_matches", 0)) for lv in levels]
        min_match = [int(lv.get("min_matches_required", 0)) for lv in levels]
        n_inliers = [int(lv.get("n_inliers", 0)) for lv in levels]

        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, n_match, width, label="mutual NN")
        ax.bar(x, min_match, width, label="min required")
        ax.bar(x + width, n_inliers, width, label="best inliers")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("count")
        ax.set_title("Feature pyramid: matches vs thresholds")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        for i, lv in enumerate(levels):
            ax.text(
                i,
                max(n_match[i], min_match[i], n_inliers[i], 1) + 1,
                str(lv.get("status", "")),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=15,
            )
        fig.tight_layout()
        path = os.path.join(self.debug_dir, "pyramid_summary.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def save_feature_map_opt_summary(self, record: Dict[str, Any]):
        """Bar chart of cosine score per multistart seed / pyramid pool."""
        rows = record.get("multistart") or []
        if not rows:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"÷{r.get('pool', '?')} {r.get('seed', '')}" for r in rows]
        cosines = [float(r.get("cosine", 0.0)) for r in rows]
        inits = [float(r.get("init_cosine", r.get("cosine", 0.0))) for r in rows]
        angles = [float(r.get("angle_deg", 0.0)) for r in rows]
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
        width = 0.38
        ax1.bar(x - width / 2, inits, width, color="lightsteelblue", alpha=0.9, label="seed cosine")
        bars = ax1.bar(x + width / 2, cosines, width, color="steelblue", alpha=0.9, label="after GD")
        ax1.axhline(
            float(record.get("cosine_identity", 0.0)),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="identity baseline",
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax1.set_ylabel("masked mean cosine")
        ax1.set_title(
            f"Feature-map multistart (best={float(record.get('cosine_opt', 0)):.3f}, "
            f"angle={float(record.get('angle_deg', record.get('accepted_angle', 0))):.1f}°)"
        )
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, axis="y", alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(x, angles, color="darkorange", marker="o", linewidth=1.2, label="angle °")
        ax2.set_ylabel("rotation angle (deg)")
        ax2.legend(loc="upper right", fontsize=8)
        for bar, c in zip(bars, cosines):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{c:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        fig.tight_layout()
        path = os.path.join(self.debug_dir, "feature_map_multistart.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def finalize(self):
        self.meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(self.debug_dir, "run_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, default=_json_default)

    def _save_rgb_image(self, rgb_hw3, mask_hw, filename, title, scales_mm=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rgb = np.asarray(rgb_hw3, dtype=np.float32)
        mask = np.asarray(mask_hw, dtype=bool)
        show = rgb.copy()
        show[~mask] = 0.0
        fh, fw = int(show.shape[0]), int(show.shape[1])
        extent = None
        aspect = "auto"
        if scales_mm is not None:
            sx, sy = float(scales_mm[0]), float(scales_mm[1])
            # rows=X tokens, cols=Y tokens → physical extent in mm
            extent = [0.0, fw * sy, 0.0, fh * sx]
            aspect = "equal"
            fig_w = 8.0
            fig_h = max(3.0, fig_w * (fh * sx) / max(fw * sy, 1e-6))
            fig_h = min(fig_h, 14.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(
            show,
            origin="lower",
            interpolation="nearest",
            extent=extent,
            aspect=aspect,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        path = os.path.join(self.debug_dir, filename)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _volume_mip(masked_volume, mask, axis=2):
    """Max-intensity projection along axis on foreground voxels."""
    vol = np.asarray(masked_volume)
    m = np.asarray(mask, dtype=bool)
    if vol.shape != m.shape:
        raise ValueError("volume and mask must share shape")
    out = np.zeros(vol.shape[:2] if axis == 2 else (vol.shape[0], vol.shape[2]), dtype=np.float32)
    if axis == 2:
        for i in range(vol.shape[0]):
            for j in range(vol.shape[1]):
                sl = vol[i, j, :]
                sm = m[i, j, :]
                out[i, j] = float(sl[sm].max()) if sm.any() else 0.0
        return out
    raise ValueError("only axis=2 MIP implemented")


def append_dino_reg_apply_forensics(
    debug_dir: str,
    *,
    pose: dict,
    reference_data,
    transformed_data,
    scan_content_mask,
    reference_content_mask,
    scan_centroid_voxel,
    reference_centroid_voxel,
    scipy_rotation,
    residual_translation_voxel,
    warp: dict,
    reference_voxel_size: float,
    scan_name: str = "",
):
    """
    After scipy volume warp, append apply-stage forensics to dinoreg-rigid-debug.

    Explains whether mis-paste (corner shift) came from centroid choice, residual
    translation, or embed indices vs the estimated physical pose.
    """
    os.makedirs(debug_dir, exist_ok=True)
    vs = float(reference_voxel_size)
    scan_c = np.asarray(scan_centroid_voxel, dtype=np.float64).reshape(3)
    ref_c = np.asarray(reference_centroid_voxel, dtype=np.float64).reshape(3)
    residual = np.asarray(residual_translation_voxel, dtype=np.float64).reshape(3)
    rotation = np.asarray(scipy_rotation, dtype=np.float64).reshape(3, 3)
    R_col = np.asarray(pose.get("R"), dtype=np.float64).reshape(3, 3)
    t_mm = np.asarray(pose.get("t"), dtype=np.float64).reshape(3)

    est_scan_mm = np.asarray(pose.get("scan_centroid_mm"), dtype=np.float64).reshape(3)
    est_ref_mm = np.asarray(pose.get("reference_centroid_mm"), dtype=np.float64).reshape(3)
    dino_vs_content_scan_vox = scan_c - est_scan_mm / vs
    dino_vs_content_ref_vox = ref_c - est_ref_mm / vs

    embed = {
        "padding_low": np.asarray(warp.get("padding_low"), dtype=np.float64).tolist(),
        "padding_high": np.asarray(warp.get("padding_high"), dtype=np.float64).tolist(),
        "rotation_scipy": rotation.tolist(),
        "offset": np.asarray(warp.get("offset"), dtype=np.float64).tolist(),
        "residual_translation_voxel": residual.tolist(),
        "padded_scan_centroid_voxel": np.asarray(warp.get("padded_scan_centroid"), dtype=np.float64).tolist(),
        "embed_centroid_voxel": np.asarray(
            warp.get("embed", {}).get("embed_centroid_voxel", warp.get("padded_scan_centroid")),
            dtype=np.float64,
        ).tolist(),
        "reference_centroid_voxel": np.asarray(warp.get("reference_centroid_voxel"), dtype=np.float64).tolist(),
        "crop_indices": warp.get("embed", {}).get("crop_indices") if isinstance(warp.get("embed"), dict) else None,
        "canvas_in": {
            "x": [int(warp.get("in_start_x", 0)), int(warp.get("in_end_x", 0))],
            "y": [int(warp.get("in_start_y", 0)), int(warp.get("in_end_y", 0))],
            "z": [int(warp.get("in_start_z", 0)), int(warp.get("in_end_z", 0))],
        },
        "volume_out": {
            "x": [int(warp.get("out_start_x", 0)), int(warp.get("out_end_x", 0))],
            "y": [int(warp.get("out_start_y", 0)), int(warp.get("out_end_y", 0))],
            "z": [int(warp.get("out_start_z", 0)), int(warp.get("out_end_z", 0))],
        },
    }

    apply_record = {
        "scan": scan_name,
        "physical_pose_mm": {
            "R_col": R_col.tolist(),
            "t_mm": t_mm.tolist(),
            "source": pose.get("source"),
            "angle_deg": float(pose.get("angle_deg", 0.0)),
        },
        "centroids_voxel": {
            "content_mask_scan": scan_c.tolist(),
            "content_mask_reference": ref_c.tolist(),
            "dino_fg_scan": (est_scan_mm / vs).tolist(),
            "dino_fg_reference": (est_ref_mm / vs).tolist(),
            "dino_minus_content_scan_voxel": dino_vs_content_scan_vox.tolist(),
            "dino_minus_content_reference_voxel": dino_vs_content_ref_vox.tolist(),
        },
        "scipy_projection": embed,
        "shapes": {
            "reference": list(reference_data.shape),
            "transformed_canvas": list(transformed_data.shape),
        },
        "notes": (
            "Estimation uses DINO adaptive-FG centroids in mm; apply uses threshold "
            "content-mask centroids (same as ANTs/GPU-rigid). Large dino_minus_content "
            "values often explain corner pastes when only the pose was correct."
        ),
    }

    apply_path = os.path.join(debug_dir, "apply_warp_forensics.json")
    with open(apply_path, "w", encoding="utf-8") as fh:
        json.dump(apply_record, fh, indent=2, default=_json_default)

    summary_path = os.path.join(debug_dir, "run_summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    summary["apply_warp"] = apply_record
    summary["apply_warp_written_utc"] = datetime.now(timezone.utc).isoformat()
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref_m = np.asarray(reference_content_mask, dtype=bool)
        mov_m = np.asarray(scan_content_mask, dtype=bool)
        ref_fg = np.asarray(reference_data, dtype=np.float32)
        mov_fg = np.asarray(transformed_data, dtype=np.float32)
        ref_mip = _volume_mip(ref_fg, ref_m)
        mov_mip = _volume_mip(mov_fg, ref_m)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(ref_mip.T, origin="lower", cmap="gray")
        axes[0].set_title("Reference content MIP (XY)")
        axes[0].scatter([ref_c[0]], [ref_c[1]], c="lime", s=30, label="content c")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(mov_mip.T, origin="lower", cmap="gray")
        axes[1].set_title("Aligned moving content MIP (XY)")
        axes[1].scatter([ref_c[0]], [ref_c[1]], c="lime", s=30, label="ref c on canvas")
        axes[1].axis("off")

        overlay = np.zeros(ref_mip.shape + (3,), dtype=np.float32)
        r = ref_mip / max(float(ref_mip.max()), 1e-6)
        m = mov_mip / max(float(mov_mip.max()), 1e-6)
        overlay[..., 0] = r
        overlay[..., 1] = m
        overlay[..., 2] = 0.25 * (r + m)
        axes[2].imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
        axes[2].set_title("Overlay R=ref G=aligned")
        axes[2].axis("off")
        fig.suptitle(f"{scan_name} apply forensics (pose source={pose.get('source')})")
        fig.tight_layout()
        fig.savefig(
            os.path.join(debug_dir, "apply_mip_overlay.png"),
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    except Exception as exc:
        with open(os.path.join(debug_dir, "apply_mip_overlay.error.txt"), "w", encoding="utf-8") as fh:
            fh.write(str(exc))
