#!/usr/bin/env python3
"""
One-shot / reusable upgrade: rewrite algorithm.math to the captioned array schema.

Schema:
  "math": [
    {"equation": "...", "caption": "symbol: meaning; ..."},
    ...
  ]

Use [] when prose alone is clearer — equations must earn their place.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "docs_index.json")

Eq = Dict[str, str]
Math = List[Eq]


def eq(equation: str, caption: str) -> Eq:
    return {"equation": equation, "caption": caption}


# None = leave unchanged (should not happen). Missing key = leave unchanged.
# Explicit [] = clear math (prose-only).
MATH_BY_ID: Dict[str, Optional[Math]] = {
    # --- Project / I/O ---
    "clear-intermediate-files": [
        eq(
            r"K = \{E_{\mathrm{elastic}},\, E_{\mathrm{elastic}}-1\} \cup \mathrm{originals}",
            r"K: files kept after cleanup; E_elastic: latest elastic-registration edit index; originals: unedited source files.",
        ),
    ],
    "extract-scans-preflight": [
        eq(
            r"\mathrm{split\ if}\ \mathrm{centroid\_size} > 1.5 \times \mathrm{median}(\mathrm{group})",
            r"centroid_size: mesh scale proxy per subject; group: subjects currently clustered together; 1.5× median: scale-warning threshold.",
        ),
    ],
    "extract-scans": [
        eq(
            r"\mathrm{voxel\_size} \ge \max\!\left(\frac{2\cdot\mathrm{mean}(\mathrm{centroid\_size})}{512},\, \frac{\max(\mathrm{extent})}{508}\right)",
            r"voxel_size: isotropic spacing written when voxelizing meshes; centroid_size / extent: subject size estimates from the mesh cohort; 512: MESH_MAX_VOXEL_CUBE; 508: usable_dim = MESH_MAX_VOXEL_CUBE − 2·MESH_VOXEL_PADDING with MESH_VOXEL_PADDING = 2 (2 voxels of padding on each side of the 512³ cube).",
        ),
    ],
    "display-nifti": [
        eq(
            r"v_{\mathrm{lossy}} = (v_{\mathrm{raw}} - z) / s",
            r"v_lossy: streamed 8-bit intensity; v_raw: full-resolution value; z: zero_point_shift; s: scale_factor from metadata.",
        ),
    ],
    "physically-accurate-mesh": [
        eq(
            r"V = \mathrm{MC}(I,\, \tau,\, \Delta)",
            r"V: mesh vertices; MC: marching cubes; I: volume; τ: iso-threshold from metadata; Δ: physical voxel spacing.",
        ),
    ],
    "get-all-subjects-mesh": [
        eq(
            r"M_{\cup} = \bigvee_i M_i",
            r"M_∪: combined foreground mask; M_i: largest connected component of subject i in reference space; ∨: voxel-wise OR.",
        ),
        eq(
            r"S = \mathrm{MC}(M_{\cup})",
            r"S: overlay surface; MC: marching cubes on the union mask.",
        ),
    ],
    "apply-rotation": [
        eq(
            r"v' = R(v - c) + c + o",
            r"v: input voxel coordinate; R: rotation matrix; c: rotation center (voxel ZYX); o: optional translation offset; v': rotated coordinate.",
        ),
    ],
    "apply-crop": [
        eq(
            r"V_{\mathrm{crop}} = \mathrm{pad}\!\left(V[z_0{:}z_1,\, y_0{:}y_1,\, x_0{:}x_1],\, p,\, b\right)",
            r"V: volume; [z0:z1, …]: crop box in ZYX; p: padding width; b: background fill value.",
        ),
    ],
    "ensure-quick-grid-projections": [],  # trivial uint8 stretch — prose covers it
    # --- Landmarks / atlas ---
    "random-landmarking": [
        eq(
            r"\max_{\mathcal{S}} \min_{i \neq j \in \mathcal{S}} \|x_i - x_j\|",
            r"S: selected landmark subset (FPS); x_i: candidate points from Poisson-disk sampling; objective: maximize nearest-neighbor separation.",
        ),
    ],
    "upload-landmarks": [
        eq(
            r"\mathrm{mismatch\ if}\ \overline{d} > k \cdot \Delta",
            r"d̄: mean landmark-to-surface distance after upload; Δ: subject voxel_size; k = LANDMARK_UPLOAD_MISMATCH_VOXEL_MULTIPLIER = 3 (mismatch when d̄ > 3·Δ).",
        ),
    ],
    "transfer-landmarks": [
        eq(
            r"x' = x + u(x)",
            r"x: source landmark; u(x): displacement sampled from the registration vector field; x': landmark on the target.",
        ),
        eq(
            r"u(x) = \mathrm{trilinear}(U,\, x)",
            r"U: dense displacement field; trilinear: interpolation over the local 3×3×3 neighborhood (with lossy shrink-factor handling).",
        ),
    ],
    "create-atlas": [
        eq(
            r"I_{\mathrm{atlas}} = \mathrm{warp}\!\left(I_{\mathrm{mean}},\, \overline{F}_{\mathrm{fwd}}^{-1}\right)",
            r"I_mean: average of rigidly aligned images; F̄_fwd: mean forward deformation; warp(·, F⁻¹): pull intensities back to the unbiased atlas frame.",
        ),
    ],
    # --- Mesh display / cleanup ---
    "marchingcubes": [
        eq(
            r"S = \{ v \mid I(v) \ge \tau \}",
            r"S: iso-surface; I(v): volume intensity at voxel v; τ: threshold (mapped through lossy params when needed).",
        ),
        eq(
            r"\Delta = \delta \cdot r",
            r"Δ: marching-cubes spacing; δ: native voxel_size; r: resolution_factor from lossy compression metadata.",
        ),
    ],
    "cleanup-mesh": [
        eq(
            r"b = \mathrm{mode}\{ v \mid v < \tau \}",
            r"b: estimated background intensity; v: subsampled voxel values; τ: threshold from metadata.",
        ),
        eq(
            r"\mathcal{I} = \{ v \mid (G_\sigma * I)(v) > \tau \}",
            r"I: island mask; G_σ: optional Gaussian smooth; σ: blur scale; τ: threshold.",
        ),
        eq(
            r"\mathrm{keep} = \arg\max_C |C|\ \mathrm{or}\ \{ C : |C| \ge p\% \cdot |F| \}",
            r"C: connected component; |C|: component size; F: foreground; p = min_island_volume_percent (default 1.0, clamped to 0.1–30) in volume-threshold mode.",
        ),
        eq(
            r"v_{\mathrm{removed}} \leftarrow b",
            r"Removed island voxels are replaced with background b.",
        ),
    ],
    # --- Preprocessing ---
    "homogenize-background": [
        eq(
            r"b = \mathrm{mode}\{ v \mid 0 < v < \tau \}",
            r"b: per-scan (or shared) background estimate; v: voxel intensity; τ: current threshold.",
        ),
        eq(
            r"v' = \mathrm{clip}(v - b)",
            r"v': background-corrected intensity, clipped to the dtype range.",
        ),
        eq(
            r"\tau' = \tau - b",
            r"τ': threshold shifted by the same background offset so segmentation stays consistent.",
        ),
    ],
    "remove-background": [
        eq(
            r"v' = \begin{cases} \min(I) & v < \tau \\ v & \mathrm{otherwise} \end{cases}",
            r"v: input intensity; τ: threshold; values below τ are floored to the volume minimum (effectively cleared).",
        ),
    ],
    "save-background-values": [
        eq(
            r"v' = \mathrm{clip}(v - b_s)",
            r"b_s: saved per-scan background; v': corrected intensity clipped to dtype bounds.",
        ),
        eq(
            r"\tau' = \tau - b_s",
            r"Threshold is shifted with the same per-scan background.",
        ),
    ],
    "apply-threshold": [
        eq(
            r"T_{\mathrm{scan}} = T_{\mathrm{ref}}",
            r"same-threshold: copy the reference threshold onto each target scan.",
        ),
        eq(
            r"T_{\mathrm{scan}} = P_{\mathrm{scan}}\!\left(\mathrm{rank}(T_{\mathrm{ref}}; I_{\mathrm{ref}})\right)",
            r"percentile-matching: map the reference threshold’s percentile rank onto the target intensity distribution P_scan.",
        ),
        eq(
            r"T_{\mathrm{scan}} = T_{\mathrm{ref}} - (p_{\mathrm{ref}} - p_{\mathrm{scan}})",
            r"histogram-matching: shift by the difference of intensity histogram peaks p_ref and p_scan.",
        ),
        eq(
            r"T^\star = \arg\max_T \left( H_{\mathrm{bg}}(T) + H_{\mathrm{fg}}(T) \right)",
            r"Kapur: choose T maximizing background + foreground Shannon entropy.",
        ),
    ],
    "match-histogram": [
        eq(
            r"v' = \mathrm{clip}(v + T_{\mathrm{ref}} - T_{\mathrm{scan}})",
            r"peak_align: translate intensities so the scan threshold/peak lines up with the reference.",
        ),
        eq(
            r"v' = \frac{\mathrm{clip}(v, p_L, p_H) - p_L}{p_H - p_L}\,(M - m) + m",
            r"percentile_normalize: stretch the [p_L, p_H] percentile window onto [m, M] (typically dtype min/max).",
        ),
    ],
    "n4-bias-correction": [
        eq(
            r"I_{\mathrm{corr}}(x) = I(x) / B(x)",
            r"I: observed intensity; B: smooth bias field from iterative B-spline N4 fitting; x: voxel; I_corr: corrected image.",
        ),
    ],
    "save-threshold-values": [],  # trivial metadata write
    "get-rotation-matrix": [],  # trivial read of stored R
    # --- Denoise / restore ---
    "denoise-all-scans": [],  # library backends — use algorithm.references, not equations
    "restore-all-scans": [
        eq(
            r"\mathrm{clip\_limit} = c \cdot \overline{h}",
            r"CLAHE: c is the user clip_limit multiplier (default parameters.clip_limit = 0.01); h̄ is the mean histogram bin count in a tile; peaks above c·h̄ are redistributed before CDF mapping.",
        ),
    ],
    "gpu-nlm3d": [
        eq(
            r"w(p,q) = \exp\!\left(-\frac{\|P(p)-P(q)\|^2}{2h^2}\right)",
            r"w: patch-similarity weight; P(·): local 3D patch; h: filter strength.",
        ),
        eq(
            r"\hat{I}(p) = \frac{\sum_q w(p,q)\, I(q)}{\sum_q w(p,q)}",
            r"Denoised value at p is the normalized weighted sum over search-window voxels q.",
        ),
    ],
    # --- Masks ---
    "save-mask": [
        eq(
            r"z_i = t_i / s_i",
            r"z_i: per-axis zoom factor; t_i: target shape; s_i: source mask shape; nearest-neighbor zoom (order=0) follows.",
        ),
    ],
    "load-mask": [],  # axis swizzle + flatten — implementation detail
    "upload-mask": [
        eq(
            r"M_{\mathrm{lossy}} = \mathrm{zoom}(M,\, [L_i/F_i],\, order{=}0)",
            r"M: full-resolution mask; L_i/F_i: lossy vs full shape ratio per axis; order=0: nearest-neighbor label preservation.",
        ),
    ],
    "apply-mask-to-image": [
        eq(
            r"I' = I \cdot \mathbf{1}[M > 0]",
            r"I: image; M: mask; voxels with M≤0 are zeroed.",
        ),
    ],
    "batch-apply-mask-to-scans": [
        eq(
            r"I' = I \cdot \mathrm{dilate}(M = \ell,\, d)",
            r"ℓ: selected label; d: dilation radius in voxels; dilate expands the label before masking.",
        ),
    ],
    "mask-out-labels": [
        eq(
            r"C = \bigvee_i (M = \ell_i)",
            r"C: combined label mask; ℓ_i: selected labels; ∨: OR over label indicators.",
        ),
        eq(
            r"I' = I \cdot (C\ \mathrm{if\ invert\ else}\ \neg C)",
            r"invert: keep the labeled region; otherwise zero it out.",
        ),
    ],
    "batch-mask-out-scans": [
        eq(
            r"C = \mathrm{dilate}(M = \ell,\, d)",
            r"C: dilated label region; ℓ: label; d: dilation.",
        ),
        eq(
            r"I'(v) = 0\ \mathrm{where}\ (C\ \mathrm{xor\ invert})(v)",
            r"Zero voxels inside C (or outside C when invert is set).",
        ),
    ],
    # --- Segmentation / edges / smart LM ---
    "run-ai-segmentation": [],  # mask zoom bookkeeping — not insightful
    "medsam2-segmentation": [
        eq(
            r"L = \{(d,h,w) \mid M_{dhw} = \ell\}",
            r"L: voxels of the prompt label ℓ in mask M.",
        ),
        eq(
            r"b_{\min} = \min L",
            r"Lower corner of the axis-aligned bounding box over label voxels L.",
        ),
        eq(
            r"b_{\max} = \max L",
            r"Upper corner of the box used as the MedSAM2 spatial prompt.",
        ),
    ],
    "edge-detection-slice-preview": [
        eq(
            r"\hat{v} = (v - v_{\min}) / (v_{\max} - v_{\min})",
            r"Per-slice min–max normalization before the edge detector.",
        ),
        eq(
            r"\mathrm{keep}\ \hat{v} \ge t_{\mathrm{high}}\ \mathrm{or\ connected\ through}\ \hat{v} \ge t_{\mathrm{low}}",
            r"Hysteresis: strong edges (≥ t_high) plus weaker edges (≥ t_low) connected to them.",
        ),
    ],
    "edge-detection": [
        eq(
            r"r' = r \cdot f",
            r"r: radius (or length) measured on the lossy preview; f: resolution_factor; r': full-resolution length.",
        ),
        eq(
            r"s' = s \cdot f^{3}",
            r"s: volume measured on lossy voxels; s': physical/full-res scaled volume (cubic in f).",
        ),
    ],
    # --- Resample / interpolate ---
    "deformation-based-mri-interpolator": [
        eq(
            r"f = s_{\mathrm{thick}} / s_{\mathrm{iso}}",
            r"f: scale_factor from thick-slice spacing to isotropic spacing.",
        ),
        eq(
            r"N' = \mathrm{round}(N \cdot f)",
            r"N: original slice count along the thick axis; N': interpolated slice count.",
        ),
        eq(
            r"w_A = (p - p_A)/(p_B - p_A)",
            r"Blend weight for neighbor A at physical position p between slices p_A and p_B; w_B = 1 − w_A.",
        ),
    ],
    "interpolate-anisotropic-to-isotropic": [
        eq(
            r"p_i = i \cdot s_{\mathrm{axis}}",
            r"p_i: physical coordinate of slice i; s_axis: spacing along the anisotropic axis.",
        ),
        eq(
            r"I(p) = w_B A_{\mathrm{moved}} + w_A B_{\mathrm{moved}}",
            r"Deformation-compensated blend of neighboring slices A and B at position p.",
        ),
    ],
    # --- Mesh edits ---
    "apply-mesh-crop": [
        eq(
            r"m = \min(c_{0:3}, c_{3:6}) - p",
            r"m: padded lower corner; c: crop-box corners; p: padding.",
        ),
        eq(
            r"M = \max(c_{0:3}, c_{3:6}) + p",
            r"M: padded upper corner of the retained axis-aligned bounds.",
        ),
        eq(
            r"v' = v - m + p",
            r"After clipping, vertices are recentered into the padded crop frame.",
        ),
    ],
    "apply-mesh-slice": [
        eq(
            r"p \in W \iff \forall i:\ (p - o_i)\cdot n_i \ge 0",
            r"W: wedge kept by the slice; o_i, n_i: origin and inward normal of plane i.",
        ),
    ],
    "apply-mesh-rotation": [
        eq(
            r"v' = (v - c) R + c",
            r"v: vertex; c: rotation_center from quick_mesh_properties; R: raw rotation matrix.",
        ),
    ],
    "apply-mesh-cleanup": [
        eq(
            r"a_{\min} = (p/100)\, \sum_i a_i",
            r"a_min: area cutoff; p = min_island_volume_percent (default 1.0, clamped to 0.1–30); a_i: component surface areas; smaller islands are removed. The /100 converts percent to a fraction.",
        ),
    ],
    "apply-mesh-snap-to-mesh": [
        eq(
            r"d_i = \|p_i - q_i\|_2",
            r"d_i: snap distance; p_i: source point; q_i: closest point on the target surface.",
        ),
    ],
    # --- Elastic / rigid registration ---
    "mesh-elastic-registration-mixin": [
        eq(
            r"E = w_s E_{\mathrm{stiff}} + w_n E_{\mathrm{normal}} + E_{\mathrm{data}}",
            r"NR-ICP stage energy with landmark weight w_l = 0 in this mixin path; w_s, w_n from the stage tuple.",
        ),
        eq(
            r"\mathrm{score} = 1 - \frac{1}{N}\sum_i \mathbb{I}(d_i > 6\Delta)",
            r"Surface agreement: d_i closest-point distance; Δ = voxel_size_mm; outliers beyond 6·Δ (fixed multiplier 6 in _mesh_elastic_error_threshold_mm).",
        ),
        eq(
            r"t = \bar{y} - s R \bar{x}",
            r"Umeyama similarity init from guidepoints: R, s from SVD (reflection-fixed); t aligns centroids.",
        ),
        eq(
            r"w_j \propto 1 / \max(\|p - c_j\|, \varepsilon)",
            r"IDW upsample weights from control points c_j to vertex p (default k_neighbors = 4); ε = 1e-10 distance floor.",
        ),
    ],
    "mesh-elastic-registration": [
        eq(
            r"E = w_s E_{\mathrm{stiff}} + w_l E_{\mathrm{lm}} + w_n E_{\mathrm{n}} + E_{\mathrm{data}}",
            r"NR-ICP energy: stiffness, landmark, normal, and data terms; correspondences with d > distance_threshold get zero data weight (preset cutoffs: facial 0.08 mm, complex_mesh_volume 0.12 mm, custom default 0.10 mm).",
        ),
        eq(
            r"\tau = 6\Delta",
            r"τ = 6·Δ with fixed multiplier 6 from _mesh_elastic_error_threshold_mm; Δ = voxel_size_mm (mesh-reference spacing).",
        ),
        eq(
            r"\mathrm{score} = 1 - \frac{1}{N}\sum_i \mathbb{I}(d_i > \tau)",
            r"elastic_surface_distance; elastic_error is the mean of d_i.",
        ),
        eq(
            r"w_j \propto 1 / \max(\|p - c_j\|, \varepsilon)",
            r"k-NN / IDW upsample from deformed controls c_j (default k_neighbors = 4); ε = 1e-10 floor on distance in 1/max(∥p−c_j∥, ε).",
        ),
        eq(
            r"\alpha = 0.25 \cdot \mathrm{median\, NN}(c)",
            r"α = tps_alpha_spacing_fraction · median NN spacing among controls; default fraction MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION = 0.25 (UI option tps_alpha_spacing_fraction).",
        ),
    ],
    "compute-mesh-to-reference-surface-distances": [
        eq(
            r"d_i = \min_{q \in S} \|v_i - q\|_2",
            r"Unsigned closest-point distance from vertex v_i to reference surface S.",
        ),
    ],
    "mesh-elastic-surface-metric": [
        eq(
            r"\mathrm{score} = 1 - \frac{1}{N}\sum_i \mathbb{I}(d_i > \tau)",
            r"Fraction of vertices within tolerance τ of the reference surface; τ = error_threshold_mm, typically 6·voxel_size_mm.",
        ),
        eq(
            r"\overline{d} = \frac{1}{N}\sum_i d_i",
            r"Mean closest-point distance (elastic_error).",
        ),
    ],
    # --- ALPACA ---
    "alpaca-create-landmarks-from-mesh": [
        eq(
            r"\|x_i - x_j\| \ge r\ \forall i \neq j",
            r"Poisson-disk / blue-noise sampling: minimum spacing r is implied by the requested point count (default n_landmarks = 0 → ALIGNMENT_REFERENCE_POISSON_POINTS = 5000).",
        ),
    ],
    "alpaca-align-landmarks-to-mesh": [
        eq(
            r"s = \mathrm{rms}(X)/\mathrm{rms\,scale}",
            r"Shared RMS scale puts both clouds near unit size before ICP/RANSAC.",
        ),
        eq(
            r"T:\ \tilde{X} \mapsto \tilde{Y}",
            r"Rigid or similarity 4×4 transform in normalized coordinates.",
        ),
        eq(
            r"R' = R,\quad t' = s\, t + c_t - R\, c_s",
            r"Map T back to world coordinates using source/target centroids c_s, c_t and scale s.",
        ),
    ],
    "alpaca-get-outer-mesh": [],  # visibility heuristic — prose is clearer than a forced formula
    # --- Align / elastic voxel ---
    "align-to-reference": [
        eq(
            r"z = \delta_{\mathrm{subj}} / \delta_{\mathrm{ref}}",
            r"Resample zoom: subject voxel size over reference voxel size.",
        ),
        eq(
            r"s = \mathbb{E}\|p^{\mathrm{ref}} - \bar{p}^{\mathrm{ref}}\| / \mathbb{E}\|p^{\mathrm{subj}} - \bar{p}^{\mathrm{subj}}\|",
            r"Optional ALPACA scale from mean landmark radius of reference vs subject.",
        ),
        eq(
            r"H = X_c^\top Y_c = U\Sigma V^\top,\quad R = V^\top C U^\top",
            r"Kabsch: cross-covariance of centered landmarks; C = diag(1,1,±1) fixes reflections.",
        ),
        eq(
            r"v' = (v - \bar{x}) R + \bar{y}",
            r"Apply rigid map from subject centroid x̄ to reference centroid ȳ.",
        ),
        eq(
            r"x_{\mathrm{out}} = R^\top x_{\mathrm{in}} + (c - R^\top c)",
            r"Volume rotation via scipy uses Rᵀ about padded center c.",
        ),
    ],
    "elastic-registration": [
        eq(
            r"\mathrm{Dice} = 2|A \cap B| / (|A| + |B|)",
            r"Overlap of thresholded masks A and B.",
        ),
        eq(
            r"\mathrm{score} = 1 - \frac{1}{N}\sum_i \mathbb{I}(d_i > 6\Delta)",
            r"elastic_surface_distance on mesh/surface samples; outlier cutoff 6·Δ with fixed multiplier 6 (error_threshold = 6 × voxel_size); Δ: voxel size.",
        ),
        eq(
            r"e = \mathrm{mean}|I - J|\_{\mathrm{overlap}}",
            r"elastic_error: mean absolute intensity difference on the overlapping foreground after SyN.",
        ),
    ],
    "invert-alignment-landmarks": [
        eq(
            r"v_{\mathrm{pad}} = R^\top (v_{\mathrm{c}} - c_{\mathrm{c}}) + c_{\mathrm{rot}}",
            r"Undo canvas rotation: v_c canvas voxel; c_c canvas centroid; c_rot original rotation center.",
        ),
        eq(
            r"v_{\mathrm{pre}} = (v_{\mathrm{pad}} - p_{\mathrm{low}}) / s",
            r"Remove padding_low and undo scale_match s.",
        ),
        eq(
            r"x_{\mathrm{mm}} = v_{\mathrm{pre}} \cdot \delta_{\mathrm{ref}}",
            r"Convert pre-scale voxels to millimeters with reference voxel size.",
        ),
    ],
    # --- RegistrationTools helpers ---
    "registration-tools-get-adjusted-background-value": [
        eq(
            r"b' = \frac{b - m}{M - m}(M' - m') + m'",
            r"Map background b from [m, M] into the normalized display range [m', M'].",
        ),
    ],
    "registration-tools-min-max-normalize": [
        eq(
            r"I' = \frac{I - m}{M - m}(M' - m') + m'",
            r"Linear min–max remap of image I from [m, M] onto [m', M'].",
        ),
    ],
    "registration-tools-restore-range": [
        eq(
            r"I_{\mathrm{restored}} = \frac{I - m}{M - m}(M_0 - m_0) + m_0",
            r"Undo a temporary normalization using the original intensity bounds [m_0, M_0].",
        ),
    ],
    "registration-tools-calculate-affine-field": [
        eq(
            r"u(p) = (A p + t) - p",
            r"u: displacement field on the fixed grid; A, t: affine linear part and translation; p: physical coordinate.",
        ),
    ],
    "registration-tools-transform-landmarks-by-rotation": [
        eq(
            r"v' = R(v - o)",
            r"v: landmark; o: origin_offset; R: rotation; keep v' inside the final bounding box.",
        ),
    ],
    "registration-tools-transform-landmarks-by-crop": [
        eq(
            r"v_{\mathrm{crop}} = v - (z_0, y_0, x_0) + p",
            r"Shift landmarks into the cropped frame; p: padding along each axis.",
        ),
    ],
}


def walk_methods(node: Any, visitor) -> None:
    if isinstance(node, dict):
        if "algorithm" in node and isinstance(node.get("algorithm"), dict) and "id" in node:
            visitor(node)
        for v in node.values():
            walk_methods(v, visitor)
    elif isinstance(node, list):
        for v in node:
            walk_methods(v, visitor)


def main() -> None:
    with open(_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    cleared = 0
    missing = []

    def visit(method: Dict[str, Any]) -> None:
        nonlocal updated, cleared
        mid = method.get("id")
        if mid not in MATH_BY_ID:
            if method.get("algorithm", {}).get("math"):
                missing.append(mid)
            return
        new_math = MATH_BY_ID[mid]
        if new_math is None:
            return
        method.setdefault("algorithm", {})
        method["algorithm"]["math"] = new_math
        if new_math:
            updated += 1
        else:
            cleared += 1

    walk_methods(data, visit)

    with open(_INDEX, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated with equations: {updated}")
    print(f"Cleared (prose-only): {cleared}")
    if missing:
        print(f"Entries still with math but not in map ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
