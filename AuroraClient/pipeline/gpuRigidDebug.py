"""
Debug checkpoints for GPU Rigid (FireANTs) alignment.

Writes marching-cubes mesh overlays (reference + warped moving) under
``extracted/{scan}/gpu-rigid-debug/`` so each pipeline stage can be compared visually.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure

DEBUG_SUBDIR_NAME = "gpu-rigid-debug"

# Normalized ANTs estimation volumes use background -1; foreground is roughly [0, 1].
DEFAULT_ISO_LEVEL = -0.5


def resolve_gpu_rigid_debug_dir(project_directory: str, scan_name: str) -> str:
    return os.path.join(project_directory, "extracted", scan_name, DEBUG_SUBDIR_NAME)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def _spacing_xyz(ants_image) -> Tuple[float, float, float]:
    spacing = getattr(ants_image, "spacing", None) or (1.0, 1.0, 1.0)
    return tuple(float(s) for s in spacing)


def _extract_mesh(volume, spacing, iso_level=DEFAULT_ISO_LEVEL, step_size=2):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.size == 0:
        return None, None
    vol = ndimage.gaussian_filter(vol, sigma=0.5)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            vol,
            level=float(iso_level),
            spacing=spacing,
            step_size=max(1, int(step_size)),
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None, None
    if len(faces) == 0:
        return None, None
    faces = faces[:, ::-1]
    return verts, faces


def _write_ants_rigid_mat(R, t, out_path=None):
    """Persist y = R x + t as an ANTs AffineTransform .mat (rotation+translation only)."""
    import ants

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mat")
        os.close(fd)
    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        precision="float",
        dimension=3,
        matrix=np.asarray(R, dtype=np.float64),
        translation=np.asarray(t, dtype=np.float64).tolist(),
        center=(0.0, 0.0, 0.0),
    )
    ants.write_transform(tx, out_path)
    return out_path


def _warp_moving_volume(fixed_image, moving_image, R: np.ndarray, t: np.ndarray):
    """Apply moving→fixed rigid transform on the estimation grid (same path as scoring)."""
    import ants

    tx_path = _write_ants_rigid_mat(R, t)
    try:
        warped = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=[tx_path],
            interpolator="linear",
            defaultvalue=-1,
            verbose=False,
        )
        return np.asarray(warped.numpy(), dtype=np.float32)
    finally:
        try:
            os.remove(tx_path)
        except OSError:
            pass


def _mesh_axis_limits(fixed_verts, moving_verts):
    if fixed_verts is None and moving_verts is None:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    pts = []
    if fixed_verts is not None and len(fixed_verts):
        pts.append(np.asarray(fixed_verts, dtype=np.float64))
    if moving_verts is not None and len(moving_verts):
        pts.append(np.asarray(moving_verts, dtype=np.float64))
    all_pts = np.vstack(pts)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-3)
    return (
        (float(lo[0] - pad[0]), float(hi[0] + pad[0])),
        (float(lo[1] - pad[1]), float(hi[1] + pad[1])),
        (float(lo[2] - pad[2]), float(hi[2] + pad[2])),
    )


def _plot_mesh_overlay(
    fixed_verts,
    fixed_faces,
    moving_verts,
    moving_faces,
    title: str,
    subtitle: str = "",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlim, ylim, zlim = _mesh_axis_limits(fixed_verts, moving_verts)
    views = [
        ("iso", {"elev": 25, "azim": -60}),
        ("xy", {"elev": 90, "azim": -90}),
        ("xz", {"elev": 0, "azim": -90}),
        ("yz", {"elev": 0, "azim": 0}),
    ]
    fig = plt.figure(figsize=(14, 11), facecolor="white")
    fig.suptitle(title, fontsize=12, y=0.98)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=9, color="#444444")

    for idx, (name, view) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        if fixed_verts is not None and fixed_faces is not None:
            ax.plot_trisurf(
                fixed_verts[:, 0],
                fixed_verts[:, 1],
                fixed_faces,
                fixed_verts[:, 2],
                color="#ff8c42",
                alpha=0.45,
                edgecolor="none",
                linewidth=0,
            )
        if moving_verts is not None and moving_faces is not None:
            ax.plot_trisurf(
                moving_verts[:, 0],
                moving_verts[:, 1],
                moving_faces,
                moving_verts[:, 2],
                color="#4fc3f7",
                alpha=0.45,
                edgecolor="none",
                linewidth=0,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect(
            (
                max(xlim[1] - xlim[0], 1e-6),
                max(ylim[1] - ylim[0], 1e-6),
                max(zlim[1] - zlim[0], 1e-6),
            )
        )
        ax.view_init(**view)
        ax.set_title(name.upper(), fontsize=10)
        ax.set_axis_off()

    fig.text(
        0.5,
        0.02,
        "Orange = reference (fixed)   |   Cyan = moving (warped to fixed space)",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


class GpuRigidDebugWriter:
    """Collect mesh-overlay PNGs + JSON forensics for one GPU Rigid run."""

    def __init__(
        self,
        debug_dir: str,
        scan_name: str = "",
        reference_name: str = "",
        *,
        iso_level: float = DEFAULT_ISO_LEVEL,
        step_size: int = 2,
    ):
        self.debug_dir = debug_dir
        self.scan_name = scan_name or "scan"
        self.reference_name = reference_name or "reference"
        self.iso_level = float(iso_level)
        self.step_size = max(1, int(step_size))
        self._seq = 0
        os.makedirs(debug_dir, exist_ok=True)
        self.meta: Dict[str, Any] = {
            "scan": self.scan_name,
            "reference": self.reference_name,
            "debug_dir": debug_dir,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "iso_level": self.iso_level,
            "marching_cubes_step_size": self.step_size,
            "checkpoints": [],
        }
        readme = os.path.join(debug_dir, "00_README.txt")
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(
                "GPU Rigid (FireANTs) debug checkpoints\n"
                "----------------------------------------\n"
                "Each PNG overlays marching-cubes meshes on the estimation grid:\n"
                "  orange  = reference (fixed)\n"
                "  cyan    = moving scan warped into fixed space at that step\n\n"
                "Files are ordered by prefix (01_, 02_, …).\n"
                "Typical run: identity (paste) → moments → refine (if better) → final.\n"
                "Best pose is chosen by fixed_recall, then IoU, then NCC on the estimation grid.\n\n"
                "Numeric metrics (ncc, iou, rotation) are in run_summary.json.\n"
            )

    def set_run_context(self, **kwargs):
        self.meta["run"] = kwargs

    def record_checkpoint(
        self,
        fixed_image,
        moving_image,
        R: np.ndarray,
        t: np.ndarray,
        *,
        stage: str,
        initializer: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        direction: str = "",
        notes: str = "",
    ) -> str:
        """
        Warp moving with (R,t), extract meshes, save a 4-view overlay PNG.

        Returns the saved filename (basename).
        """
        import matplotlib.pyplot as plt

        self._seq += 1
        spacing = _spacing_xyz(fixed_image)
        fixed_vol = np.asarray(fixed_image.numpy(), dtype=np.float32)
        warped_vol = _warp_moving_volume(fixed_image, moving_image, R, t)

        fixed_verts, fixed_faces = _extract_mesh(
            fixed_vol, spacing, iso_level=self.iso_level, step_size=self.step_size
        )
        moving_verts, moving_faces = _extract_mesh(
            warped_vol, spacing, iso_level=self.iso_level, step_size=self.step_size
        )

        init_slug = initializer or "na"
        stage_slug = stage or "step"
        basename = f"{self._seq:02d}_{init_slug}_{stage_slug}.png"
        angle_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        )
        metrics = dict(metrics or {})
        subtitle_parts = [
            f"init={init_slug}",
            f"stage={stage_slug}",
            f"rot≈{angle_deg:.1f}°",
        ]
        if direction:
            subtitle_parts.append(f"dir={direction}")
        for key in ("ncc", "ref_ncc", "iou", "fixed_recall"):
            if key in metrics and metrics[key] is not None:
                subtitle_parts.append(f"{key}={float(metrics[key]):.3f}")
        if notes:
            subtitle_parts.append(notes)

        title = f"{self.scan_name} → {self.reference_name}  [{stage_slug}]"
        fig = _plot_mesh_overlay(
            fixed_verts,
            fixed_faces,
            moving_verts,
            moving_faces,
            title=title,
            subtitle="  |  ".join(subtitle_parts),
        )
        path = os.path.join(self.debug_dir, basename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        record = {
            "seq": self._seq,
            "file": basename,
            "initializer": init_slug,
            "stage": stage_slug,
            "direction": direction,
            "rotation_deg": angle_deg,
            "R": np.asarray(R, dtype=np.float64),
            "t": np.asarray(t, dtype=np.float64).reshape(3),
            "metrics": metrics,
            "notes": notes,
            "fixed_verts": int(len(fixed_verts)) if fixed_verts is not None else 0,
            "moving_verts": int(len(moving_verts)) if moving_verts is not None else 0,
        }
        self.meta["checkpoints"].append(record)
        return basename

    def save_metrics_summary_chart(self):
        checkpoints = self.meta.get("checkpoints") or []
        if len(checkpoints) < 2:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{cp['seq']:02d}\n{cp['stage'][:8]}" for cp in checkpoints]
        ious = [float((cp.get("metrics") or {}).get("iou", 0.0)) for cp in checkpoints]
        nccs = [float((cp.get("metrics") or {}).get("ncc", 0.0)) for cp in checkpoints]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        ax.bar(x - width / 2, ious, width, label="IoU", color="#ff8c42")
        ax.bar(x + width / 2, nccs, width, label="NCC", color="#4fc3f7")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("score")
        ax.set_title(f"GPU Rigid checkpoints — {self.scan_name} → {self.reference_name}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        for i, cp in enumerate(checkpoints):
            rot = cp.get("rotation_deg")
            if rot is not None:
                ax.text(i, -0.95, f"{rot:.0f}°", ha="center", fontsize=7, color="#555555")
        fig.tight_layout()
        path = os.path.join(self.debug_dir, "checkpoint_metrics_summary.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def set_outcome(self, success: bool, message: str = "", best: Optional[Dict[str, Any]] = None):
        self.meta["success"] = bool(success)
        self.meta["message"] = message
        if best is not None:
            self.meta["best_checkpoint"] = best

    def finalize(self):
        self.save_metrics_summary_chart()
        self.meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(self.debug_dir, "run_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, default=_json_default)
