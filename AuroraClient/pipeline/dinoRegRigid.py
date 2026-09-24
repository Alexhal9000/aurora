"""
DINO-Reg rigid alignment (inspired by Song et al., IEEE TMI 2025 / MICCAI 2024).

Two-pass orientation (centroid-locked, proper rotations only — no mirrors):

  Pass 1 (cheap, image-space): 24 octahedral 90° relabels of a small FG canvas
  (≈96³, axial DINO at 32×24, ~10 slices). Encode *after* posing so ViT sees
  the candidate slice geometry. Pick the cube with best masked token cosine.

  Pass 2 (expensive, once): tri-planar DINOv3 at the highest grid that fits,
  streamed into one ``(H,W,D,24)`` PCA buffer (never a full 1024-D RAM volume).
  Adam refines the winning cube. If the cube is a weak winner or refine cosine
  stays poor, multi-K cluster CoM Kabsch (the path that already worked for
  ~80% of subjects) is scored on the same cosine and refined.

  Last resort: patch-NN feature pyramid.
  Apply: ``p_fixed = R @ p_moving + t`` via the shared scipy canvas warp.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.transform import resize

from .dinoReg.loader import (
    DinoRegError,
    PATCH_SIZE,
    ensure_dino_reg_cuda,
    get_dino_model,
    get_dinov2_model,
)
from .registrationTools import RegistrationTools
REG_FEATURE_DIM = 24
SLICE_GAP = 2
# Pass 2: try the largest ViT grid first (112×96 ≈ paper's high setting).
# Smaller entries are CUDA/host OOM fallbacks only.
FEAT_GRID_CANDIDATES = ((112, 96), (96, 84), (80, 70), (48, 40))
# Pass 1 octahedral: tiny isotropic FG canvas + few axial slices + modest tokens.
OCTAHEDRAL_CANVAS = 96
OCTAHEDRAL_SLICES = 10
OCTAHEDRAL_FEAT_GRID = (32, 24)
OCTAHEDRAL_SCORE_MARGIN = 0.02
OCTAHEDRAL_MIN_COSINE = 0.12
# After Adam, below this we still try multi-K cluster CoMs.
REFINED_COSINE_CLUSTER_FALLBACK = 0.28
# Starting hint only — actual batch is probed per (feat_h, feat_w) and shrunk on OOM.
ENCODE_BATCH_SIZE = 4
ENCODE_BATCH_MAX = 32
MAX_PCA_FIT_TOKENS = 250_000
# Full FG token clouds are too large for an all-pairs similarity matrix.
# Spatial stride + a hard cap keeps matching in the same regime as voxel-patch RANSAC.
MAX_MATCH_TOKENS = 4000
MIN_MATCH_COSINE = 0.42
LOWE_RATIO = 0.80
MIN_MUTUAL_MATCHES = 15
MIN_RANSAC_INLIERS = 12
MAX_ASYMMETRIC_MATCHES = 500
# Cheap multi-scale on the *same* DINO encode (no extra ViT passes): avg-pool the
# joint-PCA grid. pool÷4 collapses an 80×70 FG to a handful of blobs — not useful.
FEATURE_PYRAMID = (2, 1)
# Moving FG noticeably smaller than reference → nested/contained alignment.
NESTED_FG_RATIO = 0.82
# Shared PCA-space clusters become one landmark each (semantic mass centres).
# Kept for the multi-K CoM Kabsch fallback (see revert note on that function).
SEMANTIC_LANDMARK_K = 12
# Multi-K is intentional: each K yields a different homologous CoM set → Kabsch pose.
SEMANTIC_LANDMARK_K_TRIES = (6, 8, 10, 12, 14, 16)
MIN_CLUSTER_TOKENS = 12
MAX_KMEANS_FIT = 80_000
MIN_LANDMARKS = 3
LANDMARK_TOKEN_STRIDE = 2
# Feature-map rigid optimizer (legacy multi-start path; not primary).
FEATURE_MAP_OPT_POOLS = (2, 1)
FEATURE_MAP_OPT_STEPS_COARSE = 100
FEATURE_MAP_OPT_STEPS_FINE = 200
FEATURE_MAP_OPT_LR = 0.06
FEATURE_MAP_OPT_MIN_COSINE_GAIN = 0.02
# Below this, PCA features are not really co-registered (identity+translation only).
FEATURE_MAP_OPT_MIN_ABSOLUTE_COSINE = 0.10
# Legacy coarse (not on the main path): mutual-NN on pooled 24-D tokens + RANSAC.
MUTUAL_NN_COARSE_POOLS = (2,)
# Adam refine after coarse Kabsch/RANSAC: pure cosine (+ weak intensity NCC).
SEMANTIC_REFINE_COARSE_POOL = 2
SEMANTIC_REFINE_COARSE_STEPS = 80
SEMANTIC_REFINE_COARSE_LR = 0.04
SEMANTIC_REFINE_FINE_POOL = 1
SEMANTIC_REFINE_FINE_STEPS = 40
SEMANTIC_REFINE_FINE_LR = 0.015
SEMANTIC_REFINE_INTENSITY_WEIGHT = 0.05
# Cheap pooled feature cosine used to rank multi-K Kabsch candidates (not inlier count).
SEMANTIC_K_RANK_POOL = 2
# Legacy octahedral helpers (kept for reference; not used on the main path).
OCTAHEDRAL_PROBE_POOL = 2
DINO_REG_DICE_WEIGHT = 0.25
OCTAHEDRAL_MIN_OVERLAP_DICE = 0.30
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_reg_tools = RegistrationTools()

ProgressCallback = Optional[Callable[[str], None]]


def _log(scan_name, message):
    prefix = f"[DINO-Reg {scan_name}]" if scan_name else "[DINO-Reg]"
    print(f"{prefix} {message}")


def _notify(progress_callback: ProgressCallback, message: str):
    if progress_callback is not None:
        progress_callback(message)


def _normalize_volume(volume):
    return _reg_tools.min_max_normalize(
        np.asarray(volume, dtype=np.float32), new_min=-1, new_max=1
    )


def _foreground_mask(normalized_volume):
    """
    Per-volume foreground for DINO encode/matching.

    Subject and reference often differ in thresholding after min-max normalize.
    We adapt per scan; skip erosion so imperfect masks still encode anatomy.
    """
    vol = np.asarray(normalized_volume, dtype=np.float32)
    p12 = float(np.percentile(vol, 12))
    p25 = float(np.percentile(vol, 25))
    floor = min(-0.30, 0.5 * p12 + 0.5 * p25)
    floor = float(np.clip(floor, -0.80, -0.20))
    mask = vol > floor
    if mask.sum() < 64:
        from .rigidAlignment import _voxel_patch_foreground_mask

        mask = _voxel_patch_foreground_mask(vol)
    else:
        mask = ndimage.binary_closing(mask, iterations=1)
    return mask


def _foreground_mask_from_raw(volume, threshold):
    """Prefer metadata ``threshold``; fall back to heuristic on normalized volume."""
    from .rigidAlignment import _content_mask_from_threshold

    if threshold is None:
        return _foreground_mask(_normalize_volume(volume))
    return _content_mask_from_threshold(volume, threshold, sigma=1.0)


def _foreground_z_extent(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[2] == 0:
        return 0, 0
    proj = mask.any(axis=(0, 1))
    if not np.any(proj):
        return 0, mask.shape[2]
    z_idx = np.flatnonzero(proj)
    return int(z_idx[0]), int(z_idx[-1]) + 1


def _encode_slice_batch(model, device, slices_bhw, input_h, input_w):
    """GPU bilinear resize + ImageNet norm + fp16 DINOv3. slices_bhw: (B, H, W) float32."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(slices_bhw)).to(device, non_blocking=True)
    x = x.unsqueeze(1)
    x = F.interpolate(x, size=(int(input_h), int(input_w)), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.forward_features(x)["x_norm_patchtokens"].float()
    return tokens.detach().cpu().numpy()


def _encode_slice_batch_gpu(model, device, slices_bhw, input_h, input_w):
    """Same as ``_encode_slice_batch`` but keeps patch tokens on GPU (B, N, C)."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(slices_bhw)).to(device, non_blocking=True)
    x = x.unsqueeze(1)
    x = F.interpolate(x, size=(int(input_h), int(input_w)), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.forward_features(x)["x_norm_patchtokens"].float()
    return tokens


def _is_cuda_oom(exc) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _shrink_encode_batch(bs, scan_name=None, label="encode"):
    """
    After a CUDA OOM: free cache and return a smaller batch size.

    Returns the new size, or ``None`` if already at 1 (caller should re-raise).
    Caller must NOT advance the slice cursor — retry the same unprocessed range.
    """
    import torch

    torch.cuda.empty_cache()
    new_bs = max(1, int(bs) // 2)
    if new_bs >= int(bs):
        return None
    _log(
        scan_name,
        f"{label}: CUDA OOM at batch={bs}, retrying unprocessed slices with batch={new_bs}",
    )
    return new_bs


def _probe_encode_batch_size(
    model,
    device,
    input_h,
    input_w,
    sample_hw,
    max_bs=ENCODE_BATCH_MAX,
    scan_name=None,
    label="encode-probe",
):
    """
    Ramp-search the largest slice batch that fits in VRAM for this input size.

    Tries 1, 2, 4, … up to ``max_bs`` (and ``max_bs`` itself if not a power of two).
    On OOM, keeps the last successful size. A failed probe never advances work —
    it only discards the trial batch.
    """
    import torch

    sample = np.asarray(sample_hw, dtype=np.float32)
    if sample.ndim != 2:
        raise ValueError("sample_hw must be a 2D slice")
    max_bs = max(1, int(max_bs))
    best = 1
    candidates = []
    c = 1
    while c <= max_bs:
        candidates.append(c)
        nxt = c * 2
        if nxt > max_bs:
            if max_bs not in candidates:
                candidates.append(max_bs)
            break
        c = nxt

    for cand in candidates:
        batch = np.stack([sample] * int(cand), axis=0)
        try:
            toks = _encode_slice_batch_gpu(model, device, batch, input_h, input_w)
            del toks
            torch.cuda.empty_cache()
            best = int(cand)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                torch.cuda.empty_cache()
                break
            raise
    _log(
        scan_name,
        f"{label}: best encode batch={best} for input {int(input_h)}x{int(input_w)} "
        f"(probed up to {max_bs})",
    )
    return int(best)


def _cached_encode_batch_size(
    model,
    device,
    feat_h,
    feat_w,
    sample_hw,
    cache,
    max_bs=ENCODE_BATCH_MAX,
    scan_name=None,
    label="encode-probe",
):
    """Probe once per (feat_h, feat_w); reuse for the rest of the run."""
    key = (int(feat_h), int(feat_w))
    if key in cache:
        return int(cache[key])
    bs = _probe_encode_batch_size(
        model,
        device,
        int(feat_h) * PATCH_SIZE,
        int(feat_w) * PATCH_SIZE,
        sample_hw,
        max_bs=max_bs,
        scan_name=scan_name,
        label=label,
    )
    cache[key] = int(bs)
    return int(bs)


def _axial_encode_indices(n_slices, gap, mask_hw_d=None):
    """Slice indices to run through DINOv2 (gap + last), optionally FG-only."""
    encode_idx = list(range(0, max(int(n_slices) - 1, 0), int(gap)))
    if n_slices > 0 and (int(n_slices) - 1) not in encode_idx:
        encode_idx.append(int(n_slices) - 1)
    if mask_hw_d is not None:
        has_fg = np.asarray(mask_hw_d, dtype=bool).any(axis=(0, 1))
        kept = [i for i in encode_idx if bool(has_fg[i])]
        if kept:
            encode_idx = kept
    return encode_idx


def _fill_encoded_gaps_hwzd(tokens_hwzd, encoded_at):
    """Linear fill along depth for slices not directly encoded (in-place)."""
    encoded_at = sorted(int(i) for i in encoded_at)
    for a, b in zip(encoded_at, encoded_at[1:]):
        span = b - a
        if span <= 1:
            continue
        fa = tokens_hwzd[:, :, a, :]
        fb = tokens_hwzd[:, :, b, :]
        for k in range(1, span):
            alpha = k / float(span)
            tokens_hwzd[:, :, a + k, :] = fa * (1.0 - alpha) + fb * alpha


def _l2_normalize_tokens_bnC(tokens, eps=1e-8):
    """L2-normalize patch tokens (B, N, C) on whatever device they live on."""
    import torch

    norms = torch.linalg.vector_norm(tokens, dim=-1, keepdim=True).clamp_min(eps)
    return tokens / norms


def _fit_pca_basis_gpu(fit_tokens_nc, n_components, device):
    """
    Fit mean + PCA basis on GPU from (N, C) float32 tokens.
    Returns (mean_1C, basis_Ck) on ``device``.
    """
    import torch

    x_fit = torch.from_numpy(np.ascontiguousarray(fit_tokens_nc, dtype=np.float32)).to(device)
    mean = x_fit.mean(dim=0)
    x_fit = x_fit - mean
    n_comp = min(int(n_components), int(x_fit.shape[0]) - 1, int(x_fit.shape[1]))
    if n_comp < 3:
        raise RuntimeError("too few tokens/components for PCA")
    q = min(n_comp + 8, x_fit.shape[0], x_fit.shape[1])
    _u, _s, v = torch.pca_lowrank(x_fit, q=q, niter=4)
    basis = v[:, :n_comp].contiguous()
    del x_fit, _u, _s, v
    return mean, basis


def _project_tokens_gpu(tokens_bnC, mean_1C, basis_Ck):
    """(B,N,C) → (B,N,k) on GPU; tokens should already be L2-normalized if desired."""
    return (tokens_bnC - mean_1C.view(1, 1, -1)) @ basis_Ck


def _sample_axial_tokens_for_pca(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap,
    mask,
    max_tokens,
    scan_name=None,
    label="sample",
    batch_size=ENCODE_BATCH_SIZE,
    rng=None,
):
    """
    Encode a sparse set of axial slices and return up to ``max_tokens`` L2-normalized
    FG patch tokens (N, C) on CPU — never allocates a full feature volume.
    """
    import torch

    volume = np.asarray(volume, dtype=np.float32)
    n_slices = int(volume.shape[2])
    input_h = int(feat_h) * PATCH_SIZE
    input_w = int(feat_w) * PATCH_SIZE
    # Coarser than match gap for the PCA fit pass (even when gap=1).
    sample_gap = max(int(gap) * 2, 2)
    encode_idx = _axial_encode_indices(n_slices, sample_gap, mask)
    if not encode_idx:
        return np.zeros((0, int(getattr(model, "embed_dim", 1024))), dtype=np.float32)

    rng = np.random.default_rng(0) if rng is None else rng
    chunks = []
    n_kept = 0
    bs = max(1, int(batch_size))
    i0 = 0
    while i0 < len(encode_idx) and n_kept < int(max_tokens):
        batch_ids = encode_idx[i0:i0 + bs]
        batch = np.stack([volume[:, :, sid] for sid in batch_ids], axis=0)
        try:
            toks = _encode_slice_batch_gpu(model, device, batch, input_h, input_w)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                new_bs = _shrink_encode_batch(bs, scan_name=scan_name, label=label)
                if new_bs is None:
                    raise
                bs = new_bs
                continue  # retry same unprocessed encode_idx[i0:]
            raise
        toks = _l2_normalize_tokens_bnC(toks)
        # toks: (B, feat_h*feat_w, C)
        b, n_pix, c = toks.shape
        flat = toks.reshape(b * n_pix, c)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            # Approximate FG at feature resolution via block resize of the slice masks.
            from skimage.transform import resize as _resize

            fg_rows = []
            for sid in batch_ids:
                sm = _resize(
                    m[:, :, sid].astype(np.float32),
                    (feat_h, feat_w),
                    anti_aliasing=False,
                    preserve_range=True,
                ) > 0.5
                fg_rows.append(sm.reshape(-1))
            fg = np.concatenate(fg_rows, axis=0)
            if fg.any():
                flat = flat[torch.from_numpy(fg).to(device)]
            else:
                i0 += len(batch_ids)
                continue
        take = min(int(flat.shape[0]), int(max_tokens) - n_kept)
        if take <= 0:
            break
        if int(flat.shape[0]) > take:
            sel = rng.choice(int(flat.shape[0]), size=take, replace=False)
            flat = flat[torch.from_numpy(sel).to(device)]
        chunks.append(flat.detach().cpu().numpy().astype(np.float32))
        n_kept += take
        del toks, flat
        i0 += len(batch_ids)
    if scan_name:
        _log(scan_name, f"{label}: sampled {n_kept} tokens for PCA fit")
    if not chunks:
        return np.zeros((0, int(getattr(model, "embed_dim", 1024))), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _encode_axial_project_into(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap,
    mask,
    mean_1C,
    basis_Ck,
    out_hwzd,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    Encode axial slices, L2-normalize, PCA-project on GPU, write into ``out_hwzd``
    of shape (feat_h, feat_w, D, k). Only one batch of 1024-D tokens exists at a time.
    """
    import torch

    volume = np.asarray(volume, dtype=np.float32)
    n_slices = int(volume.shape[2])
    assert out_hwzd.shape[:3] == (feat_h, feat_w, n_slices)
    input_h = int(feat_h) * PATCH_SIZE
    input_w = int(feat_w) * PATCH_SIZE
    encode_idx = _axial_encode_indices(n_slices, gap, mask)
    bs = max(1, int(batch_size))
    encoded_at = []
    i0 = 0
    step = 0
    while i0 < len(encode_idx):
        batch_ids = encode_idx[i0:i0 + bs]
        batch = np.stack([volume[:, :, sid] for sid in batch_ids], axis=0)
        try:
            toks = _encode_slice_batch_gpu(model, device, batch, input_h, input_w)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                new_bs = _shrink_encode_batch(bs, scan_name=scan_name, label=label)
                if new_bs is None:
                    raise
                bs = new_bs
                continue  # retry same unprocessed encode_idx[i0:] — do not skip
            raise
        toks = _l2_normalize_tokens_bnC(toks)
        proj = _project_tokens_gpu(toks, mean_1C, basis_Ck)
        proj_np = proj.detach().cpu().numpy().astype(np.float32)
        del toks, proj
        for row, sid in enumerate(batch_ids):
            out_hwzd[:, :, sid, :] = proj_np[row].reshape(feat_h, feat_w, -1)
            encoded_at.append(sid)
        step += len(batch_ids)
        if scan_name and (step <= bs or step % max(bs * 4, 8) == 0 or i0 + len(batch_ids) >= len(encode_idx)):
            _log(
                scan_name,
                f"{label} projected {len(encoded_at)}/{len(encode_idx)} slices "
                f"(last {batch_ids[-1] + 1}/{n_slices}, batch={bs})",
            )
        i0 += len(batch_ids)
    _fill_encoded_gaps_hwzd(out_hwzd, encoded_at)


def encode_volume_axial_pca_stream(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    mean_1C,
    basis_Ck,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """Axial DINOv2 → PCA-k grid ``(feat_h, feat_w, D, k)`` without a 1024-D RAM volume."""
    volume = np.asarray(volume, dtype=np.float32)
    k = int(basis_Ck.shape[1])
    out = np.zeros((feat_h, feat_w, volume.shape[2], k), dtype=np.float32)
    _encode_axial_project_into(
        volume, model, device, feat_h, feat_w, gap, mask,
        mean_1C, basis_Ck, out,
        scan_name=scan_name, label=label, batch_size=batch_size,
    )
    return out


def encode_volume_coronal_pca_stream(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    mean_1C,
    basis_Ck,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """Coronal DINOv2 → PCA-k, resized onto the axial ``(feat_h, feat_w, D, k)`` grid."""
    volume = np.asarray(volume, dtype=np.float32)
    h, w, d = volume.shape
    feat_d = _feat_depth_for_axis(feat_h, h, w, d)
    k = int(basis_Ck.shape[1])
    vol_p = np.transpose(volume, (0, 2, 1))
    mask_p = None if mask is None else np.transpose(np.asarray(mask, dtype=bool), (0, 2, 1))
    # Encode as axial over (H, D, W) → (feat_h, feat_d, W, k)
    tmp = np.zeros((feat_h, feat_d, w, k), dtype=np.float32)
    _encode_axial_project_into(
        vol_p, model, device, feat_h, feat_d, gap, mask_p,
        mean_1C, basis_Ck, tmp,
        scan_name=scan_name, label=f"{label}-cor", batch_size=batch_size,
    )
    feat = np.transpose(tmp, (0, 2, 1, 3))  # (feat_h, W, feat_d, k)
    del tmp
    return _resize_feat_hwzd(feat, feat_h, feat_w, d)


def encode_volume_sagittal_pca_stream(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    mean_1C,
    basis_Ck,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """Sagittal DINOv2 → PCA-k, resized onto the axial ``(feat_h, feat_w, D, k)`` grid."""
    volume = np.asarray(volume, dtype=np.float32)
    h, w, d = volume.shape
    feat_d = _feat_depth_for_axis(feat_h, h, w, d)
    k = int(basis_Ck.shape[1])
    vol_p = np.transpose(volume, (1, 2, 0))
    mask_p = None if mask is None else np.transpose(np.asarray(mask, dtype=bool), (1, 2, 0))
    tmp = np.zeros((feat_w, feat_d, h, k), dtype=np.float32)
    _encode_axial_project_into(
        vol_p, model, device, feat_w, feat_d, gap, mask_p,
        mean_1C, basis_Ck, tmp,
        scan_name=scan_name, label=f"{label}-sag", batch_size=batch_size,
    )
    feat = np.transpose(tmp, (2, 0, 1, 3))  # (H, feat_w, feat_d, k)
    del tmp
    return _resize_feat_hwzd(feat, feat_h, feat_w, d)


def encode_pair_triplanar_joint_pca(
    moving_volume,
    fixed_volume,
    model,
    device,
    feat_h,
    feat_w,
    moving_mask=None,
    fixed_mask=None,
    gap=SLICE_GAP,
    n_components=REG_FEATURE_DIM,
    scan_name=None,
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    Memory-efficient tri-planar encode + joint PCA for a moving/fixed pair.

    Old path allocated full 1024-D grids (``feat_h×feat_w×D×1024`` ≈ 10+ GB at
    80×70) *per plane* in host RAM. This path:
      1) samples sparse L2-normalized tokens once per subject × plane
         (moving+fixed × axial/coronal/sagittal) and fits one joint PCA basis
      2) encodes each volume once more to PCA-project into a single
         ``(H,W,D,k)`` float32 buffer (k=24)

    Batch size is ramp-probed per feature grid shape (largest that fits VRAM).
    Mid-run OOMs shrink the batch and retry the same unprocessed slices.

    Returns ``(mov_feat_hwzk, fix_feat_hwzk)``.
    """
    import gc

    import torch

    moving_volume = np.asarray(moving_volume, dtype=np.float32)
    fixed_volume = np.asarray(fixed_volume, dtype=np.float32)
    batch_cache = {}
    z_m = int(moving_volume.shape[2] // 2)
    sample_ax = moving_volume[:, :, z_m]
    bs_ax = _cached_encode_batch_size(
        model, device, feat_h, feat_w, sample_ax, batch_cache,
        max_bs=ENCODE_BATCH_MAX, scan_name=scan_name, label="probe-axial",
    )
    # Coronal/sagittal use a different token grid → probe separately.
    h, w, d = moving_volume.shape
    feat_d = _feat_depth_for_axis(feat_h, h, w, d)
    sample_cor = np.transpose(moving_volume, (0, 2, 1))[:, :, min(w // 2, w - 1)]
    bs_cor = _cached_encode_batch_size(
        model, device, feat_h, feat_d, sample_cor, batch_cache,
        max_bs=ENCODE_BATCH_MAX, scan_name=scan_name, label="probe-coronal",
    )
    sample_sag = np.transpose(moving_volume, (1, 2, 0))[:, :, min(h // 2, h - 1)]
    bs_sag = _cached_encode_batch_size(
        model, device, feat_w, feat_d, sample_sag, batch_cache,
        max_bs=ENCODE_BATCH_MAX, scan_name=scan_name, label="probe-sagittal",
    )
    _log(
        scan_name,
        f"encode batch sizes: axial={bs_ax} coronal={bs_cor} sagittal={bs_sag} "
        f"(gap={gap})",
    )
    # ``batch_size`` arg is only a floor hint if probe somehow returns 0.
    bs_ax = max(int(bs_ax), 1)
    bs_cor = max(int(bs_cor), 1)
    bs_sag = max(int(bs_sag), 1)
    del batch_size  # probed sizes win

    rng = np.random.default_rng(0)
    # One sparse harvest per subject × plane (6 views), then a single PCA fit.
    budget = max(8_000, int(MAX_PCA_FIT_TOKENS) // 6)
    samples = []
    for vol, msk, tag in (
        (moving_volume, moving_mask, "moving"),
        (fixed_volume, fixed_mask, "fixed"),
    ):
        vol = np.asarray(vol, dtype=np.float32)
        hh, ww, dd = vol.shape
        fd = _feat_depth_for_axis(feat_h, hh, ww, dd)
        samples.append(
            _sample_axial_tokens_for_pca(
                vol, model, device, feat_h, feat_w, gap, msk, budget,
                scan_name=scan_name, label=f"{tag}-ax-pca", batch_size=bs_ax, rng=rng,
            )
        )
        vol_c = np.transpose(vol, (0, 2, 1))
        msk_c = None if msk is None else np.transpose(np.asarray(msk, dtype=bool), (0, 2, 1))
        samples.append(
            _sample_axial_tokens_for_pca(
                vol_c, model, device, feat_h, fd, gap, msk_c, budget,
                scan_name=scan_name, label=f"{tag}-cor-pca", batch_size=bs_cor, rng=rng,
            )
        )
        vol_s = np.transpose(vol, (1, 2, 0))
        msk_s = None if msk is None else np.transpose(np.asarray(msk, dtype=bool), (1, 2, 0))
        samples.append(
            _sample_axial_tokens_for_pca(
                vol_s, model, device, feat_w, fd, gap, msk_s, budget,
                scan_name=scan_name, label=f"{tag}-sag-pca", batch_size=bs_sag, rng=rng,
            )
        )

    fit = np.concatenate([s for s in samples if len(s)], axis=0)
    del samples
    if len(fit) < 32:
        raise RuntimeError("too few tokens to fit joint PCA")
    if len(fit) > MAX_PCA_FIT_TOKENS:
        sel = rng.choice(len(fit), size=MAX_PCA_FIT_TOKENS, replace=False)
        fit = fit[sel]
    _log(scan_name, f"fitting joint PCA on {len(fit)} streamed tokens → k={n_components}")
    mean_1C, basis_Ck = _fit_pca_basis_gpu(fit, n_components, device)
    del fit
    gc.collect()

    def _fuse_one(volume, mask, tag):
        ax = encode_volume_axial_pca_stream(
            volume, model, device, feat_h, feat_w, mean_1C, basis_Ck,
            gap=gap, mask=mask, scan_name=scan_name, label=f"{tag}-ax",
            batch_size=bs_ax,
        )
        co = encode_volume_coronal_pca_stream(
            volume, model, device, feat_h, feat_w, mean_1C, basis_Ck,
            gap=gap, mask=mask, scan_name=scan_name, label=tag,
            batch_size=bs_cor,
        )
        ax += co
        del co
        sg = encode_volume_sagittal_pca_stream(
            volume, model, device, feat_h, feat_w, mean_1C, basis_Ck,
            gap=gap, mask=mask, scan_name=scan_name, label=tag,
            batch_size=bs_sag,
        )
        ax += sg
        del sg
        ax *= (1.0 / 3.0)
        return ax

    mov_feat = _fuse_one(moving_volume, moving_mask, "moving")
    fix_feat = _fuse_one(fixed_volume, fixed_mask, "fixed")
    del mean_1C, basis_Ck
    torch.cuda.empty_cache()
    return mov_feat, fix_feat


def encode_volume_axial(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    GitHub ``encode_3D_gap``, but resize/encode on GPU in batches.

    Returns flattened tokens of shape (feat_h * feat_w * D, embed_dim).
    """
    import torch

    volume = np.asarray(volume, dtype=np.float32)
    n_slices = int(volume.shape[2])
    input_h = int(feat_h) * PATCH_SIZE
    input_w = int(feat_w) * PATCH_SIZE
    embed_dim = int(getattr(model, "embed_dim", 1024))
    tokens = np.zeros((feat_h * feat_w, n_slices, embed_dim), dtype=np.float32)

    encode_idx = list(range(0, max(n_slices - 1, 0), int(gap)))
    if n_slices > 0 and (n_slices - 1) not in encode_idx:
        encode_idx.append(n_slices - 1)
    if mask is not None:
        has_fg = np.asarray(mask, dtype=bool).any(axis=(0, 1))
        kept = [i for i in encode_idx if bool(has_fg[i])]
        if kept:
            encode_idx = kept

    bs = max(1, int(batch_size))
    encoded_at = []
    step = 0
    i0 = 0
    while i0 < len(encode_idx):
        batch_ids = encode_idx[i0:i0 + bs]
        batch = np.stack([volume[:, :, sid] for sid in batch_ids], axis=0)
        try:
            feats = _encode_slice_batch(model, device, batch, input_h, input_w)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                new_bs = _shrink_encode_batch(bs, scan_name=scan_name, label=label)
                if new_bs is None:
                    raise
                bs = new_bs
                continue  # retry same unprocessed slices
            raise
        for row, sid in enumerate(batch_ids):
            tokens[:, sid, :] = feats[row]
            encoded_at.append(sid)
        step += len(batch_ids)
        if scan_name and (step <= bs or step % max(bs * 4, 8) == 0 or i0 + len(batch_ids) >= len(encode_idx)):
            _log(
                scan_name,
                f"{label} encoded {len(encoded_at)}/{len(encode_idx)} axial slices "
                f"(last {batch_ids[-1] + 1}/{n_slices}, batch={bs})",
            )
        i0 += len(batch_ids)

    encoded_at.sort()
    for a, b in zip(encoded_at, encoded_at[1:]):
        span = b - a
        if span <= 1:
            continue
        fa = tokens[:, a, :]
        fb = tokens[:, b, :]
        for k in range(1, span):
            alpha = k / float(span)
            tokens[:, a + k, :] = fa * (1.0 - alpha) + fb * alpha

    return tokens.reshape(feat_h * feat_w * n_slices, embed_dim)


def _feat_depth_for_axis(feat_h, vol_h, vol_w, vol_d):
    """Feature count along a secondary volume axis (keeps PATCH-sized inputs sane)."""
    denom = max(int(vol_h), int(vol_w), 1)
    return max(8, int(round(float(feat_h) * float(vol_d) / float(denom))))


def _resize_feat_hwzd(feat, out_h, out_w, out_d):
    """Resize (H, W, D, C) feature grid with per-channel linear interpolation."""
    feat = np.asarray(feat, dtype=np.float32)
    oh, ow, od = int(out_h), int(out_w), int(out_d)
    if feat.shape[0] == oh and feat.shape[1] == ow and feat.shape[2] == od:
        return feat
    c = int(feat.shape[3])
    out = np.empty((oh, ow, od, c), dtype=np.float32)
    for ci in range(c):
        out[..., ci] = resize(
            feat[..., ci],
            (oh, ow, od),
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32)
    return out


def _l2_normalize_feat_hwzd(feat, eps=1e-8):
    feat = np.asarray(feat, dtype=np.float32)
    norms = np.linalg.norm(feat, axis=-1, keepdims=True)
    return feat / np.maximum(norms, eps)


def encode_volume_coronal(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    Coronal DINOv2: treat Y as the slice axis, map tokens onto (feat_h, feat_w, D).

    Volume layout is (H, W, D). Each coronal slice is (H, D); after encoding we
    interpolate onto the same axial feature grid used by ``encode_volume_axial``.
    """
    volume = np.asarray(volume, dtype=np.float32)
    h, w, d = volume.shape
    feat_d = _feat_depth_for_axis(feat_h, h, w, d)
    # (H, D, W) so last axis is Y — axial encoder walks coronal planes.
    vol_p = np.transpose(volume, (0, 2, 1))
    mask_p = None if mask is None else np.transpose(np.asarray(mask, dtype=bool), (0, 2, 1))
    flat = encode_volume_axial(
        vol_p,
        model,
        device,
        feat_h,
        feat_d,
        gap=gap,
        mask=mask_p,
        scan_name=scan_name,
        label=f"{label}-cor",
        batch_size=batch_size,
    )
    # (feat_h, feat_d, W, C) → (feat_h, W, feat_d, C) → resize to (feat_h, feat_w, D, C)
    feat = flat.reshape(feat_h, feat_d, w, -1)
    feat = np.transpose(feat, (0, 2, 1, 3))
    feat = _resize_feat_hwzd(feat, feat_h, feat_w, d)
    return feat.reshape(feat_h * feat_w * d, feat.shape[-1])


def encode_volume_sagittal(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    Sagittal DINOv2: treat X as the slice axis, map tokens onto (feat_h, feat_w, D).
    """
    volume = np.asarray(volume, dtype=np.float32)
    h, w, d = volume.shape
    feat_d = _feat_depth_for_axis(feat_h, h, w, d)
    # (W, D, H) so last axis is X — axial encoder walks sagittal planes.
    vol_p = np.transpose(volume, (1, 2, 0))
    mask_p = None if mask is None else np.transpose(np.asarray(mask, dtype=bool), (1, 2, 0))
    flat = encode_volume_axial(
        vol_p,
        model,
        device,
        feat_w,
        feat_d,
        gap=gap,
        mask=mask_p,
        scan_name=scan_name,
        label=f"{label}-sag",
        batch_size=batch_size,
    )
    # (feat_w, feat_d, H, C) → (H, feat_w, feat_d, C) → resize to (feat_h, feat_w, D, C)
    feat = flat.reshape(feat_w, feat_d, h, -1)
    feat = np.transpose(feat, (2, 0, 1, 3))
    feat = _resize_feat_hwzd(feat, feat_h, feat_w, d)
    return feat.reshape(feat_h * feat_w * d, feat.shape[-1])


def encode_volume_triplanar(
    volume,
    model,
    device,
    feat_h,
    feat_w,
    gap=SLICE_GAP,
    mask=None,
    scan_name=None,
    label="volume",
    batch_size=ENCODE_BATCH_SIZE,
):
    """
    Axial + coronal + sagittal DINOv2, mean-fused after per-view L2 normalize.

    Returns flattened tokens ``(feat_h * feat_w * D, C)`` on the axial feature grid
    (native depth D), matching ``encode_volume_axial``.
    """
    volume = np.asarray(volume, dtype=np.float32)
    d = int(volume.shape[2])
    axial = encode_volume_axial(
        volume, model, device, feat_h, feat_w, gap=gap, mask=mask,
        scan_name=scan_name, label=f"{label}-ax", batch_size=batch_size,
    )
    coronal = encode_volume_coronal(
        volume, model, device, feat_h, feat_w, gap=gap, mask=mask,
        scan_name=scan_name, label=label, batch_size=batch_size,
    )
    sagittal = encode_volume_sagittal(
        volume, model, device, feat_h, feat_w, gap=gap, mask=mask,
        scan_name=scan_name, label=label, batch_size=batch_size,
    )
    c = int(axial.shape[-1])
    a = _l2_normalize_feat_hwzd(axial.reshape(feat_h, feat_w, d, c))
    co = _l2_normalize_feat_hwzd(coronal.reshape(feat_h, feat_w, d, c))
    s = _l2_normalize_feat_hwzd(sagittal.reshape(feat_h, feat_w, d, c))
    fused = (a + co + s) * (1.0 / 3.0)
    return fused.reshape(feat_h * feat_w * d, c)


def _resize_mask_to_feat(mask, feat_h, feat_w):
    mask = np.asarray(mask, dtype=np.float32)
    resized = resize(
        mask,
        (feat_h, feat_w, mask.shape[2]),
        anti_aliasing=False,
        preserve_range=True,
    )
    return resized > 0.5


def _joint_pca_scatter(
    moving_tokens,
    fixed_tokens,
    moving_mask_feat,
    fixed_mask_feat,
    n_components=REG_FEATURE_DIM,
):
    """
    Paper joint PCA: flatten both volumes, keep foreground tokens, concatenate,
    low-rank PCA to k=24, scatter back onto the feature grids (background = 0).
    """
    import torch

    moving_mask = np.asarray(moving_mask_feat, dtype=bool).reshape(-1)
    fixed_mask = np.asarray(fixed_mask_feat, dtype=bool).reshape(-1)
    mov_fg = np.asarray(moving_tokens, dtype=np.float32)[moving_mask]
    fix_fg = np.asarray(fixed_tokens, dtype=np.float32)[fixed_mask]
    if len(mov_fg) < 8 or len(fix_fg) < 8:
        return None, None

    all_feat = np.concatenate([mov_fg, fix_fg], axis=0)
    n_comp = min(int(n_components), all_feat.shape[0] - 1, all_feat.shape[1])
    if n_comp < 3:
        return None, None

    rng = np.random.default_rng(0)
    if all_feat.shape[0] > MAX_PCA_FIT_TOKENS:
        fit_idx = rng.choice(all_feat.shape[0], size=MAX_PCA_FIT_TOKENS, replace=False)
        fit = all_feat[fit_idx]
    else:
        fit = all_feat

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    x_fit = torch.from_numpy(np.ascontiguousarray(fit)).to(device)
    mean = x_fit.mean(dim=0)
    x_fit = x_fit - mean
    q = min(n_comp + 8, x_fit.shape[0], x_fit.shape[1])
    _u, _s, v = torch.pca_lowrank(x_fit, q=q, niter=4)
    basis = v[:, :n_comp]
    del x_fit
    chunks = []
    chunk = 65536
    for start in range(0, all_feat.shape[0], chunk):
        x = torch.from_numpy(np.ascontiguousarray(all_feat[start:start + chunk])).to(device)
        chunks.append(((x - mean) @ basis).detach().cpu().numpy().astype(np.float32))
        del x
    reduced = np.concatenate(chunks, axis=0)

    n_mov = int(moving_mask.sum())
    mov_red = reduced[:n_mov]
    fix_red = reduced[n_mov:]

    mov_out = np.zeros((moving_tokens.shape[0], n_comp), dtype=np.float32)
    fix_out = np.zeros((fixed_tokens.shape[0], n_comp), dtype=np.float32)
    mov_out[moving_mask] = mov_red
    fix_out[fixed_mask] = fix_red
    return mov_out, fix_out


def _feat_scales_mm(shape_xyz, feat_shape, voxel_size, z0):
    """mm per feature voxel along (x, y, z), plus origin of the cropped volume."""
    h, w, d = shape_xyz
    fh, fw, fd = feat_shape
    vs = float(voxel_size)
    scales = np.array(
        [vs * h / float(fh), vs * w / float(fw), vs * d / float(fd)],
        dtype=np.float64,
    )
    origin = np.array([0.0, 0.0, float(z0) * vs], dtype=np.float64)
    return scales, origin


def _mask_centroid_mm(mask, voxel_size, z0=0):
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coords) == 0:
        shape = np.asarray(mask.shape, dtype=np.float64)
        coords = (shape - 1.0) * 0.5
        return coords * float(voxel_size) + np.array([0.0, 0.0, float(z0) * float(voxel_size)])
    c = coords.mean(axis=0).astype(np.float64)
    c[2] += float(z0)
    return c * float(voxel_size)


def _foreground_diagonal_mm(mask, voxel_size, z0=0):
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coords) < 2:
        return float(max(np.asarray(mask.shape)) * float(voxel_size))
    coords = coords.astype(np.float64)
    coords[:, 2] += float(z0)
    coords *= float(voxel_size)
    return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))


def _is_nested_foreground(mov_diag_mm, fix_diag_mm):
    """True when the moving blob is substantially smaller than the reference."""
    return float(mov_diag_mm) < float(NESTED_FG_RATIO) * float(fix_diag_mm)


def _rotation_180_about_axis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return None
    axis = axis / norm
    return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)


def _rt_to_guidepoint_pairs(R, t, moving_centroid_mm):
    """
    Encode y = R x + t as homologous mm points for the shared Kabsch warp.

    The ±axis cloud is centred on the mask centroid so the warp paste matches
    manual guidepoints / ALPACA (rotate about mean, integer centroid snap).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    c_m = np.asarray(moving_centroid_mm, dtype=np.float64)
    c_f = R @ c_m + t
    spread = max(8.0, 0.08 * (np.linalg.norm(c_m) + 1.0))
    axes = np.eye(3, dtype=np.float64) * spread
    offsets = np.vstack([axes, -axes])
    moving_pts = c_m + offsets
    fixed_pts = c_f + (offsets @ R.T)
    return moving_pts, fixed_pts


def _token_stride(n_fg, max_tokens=MAX_MATCH_TOKENS):
    """Grid stride so ~n_fg / stride^3 tokens remain, at least 1."""
    n_fg = max(int(n_fg), 1)
    max_tokens = max(int(max_tokens), 16)
    if n_fg <= max_tokens:
        return 1
    return max(1, int(np.ceil((n_fg / float(max_tokens)) ** (1.0 / 3.0))))


def _l2_normalize_rows(features, eps=1e-8):
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def _foreground_token_cloud(feat_hwzc, mask, scales_mm, origin_mm, stride=1):
    """Foreground PCA tokens and their numpy-mm centres (same frame as guidepoints)."""
    feat = np.asarray(feat_hwzc, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    s = max(1, int(stride))
    ii, jj, kk = np.nonzero(mask[::s, ::s, ::s])
    if ii.size == 0:
        empty_f = np.empty((0, feat.shape[-1]), dtype=np.float32)
        empty_p = np.empty((0, 3), dtype=np.float64)
        return empty_f, empty_p
    i = ii * s
    j = jj * s
    k = kk * s
    tokens = feat[i, j, k]
    scales = np.asarray(scales_mm, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64)
    pts = np.stack(
        [
            origin[0] + (i.astype(np.float64) + 0.5) * scales[0],
            origin[1] + (j.astype(np.float64) + 0.5) * scales[1],
            origin[2] + (k.astype(np.float64) + 0.5) * scales[2],
        ],
        axis=1,
    )
    return tokens, pts


def _cap_token_cloud(tokens, pts, max_n, rng=None):
    """Keep a spatially spread subset (deterministic linspace, not random)."""
    n = int(pts.shape[0])
    if n <= int(max_n):
        return tokens, pts
    idx = np.linspace(0, n - 1, int(max_n), dtype=np.int64)
    return tokens[idx], pts[idx]


def _scatter_labels_to_grid(shape_hwz, ijk, labels):
    grid = np.full(shape_hwz, -1, dtype=np.int32)
    if ijk.size == 0:
        return grid
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = labels.astype(np.int32)
    return grid


def _adaptive_min_cluster_tokens(k, n_mov, n_fix):
    """Size-mismatched scans leave some materials mostly on one side — stay lenient."""
    k = max(1, int(k))
    per_side = int(min(int(n_mov), int(n_fix)) / max(k * 3, 1))
    return max(8, min(int(MIN_CLUSTER_TOKENS), per_side))


def _fit_reference_semantic_clusters(mov_tok, fix_tok, k, max_fit=MAX_KMEANS_FIT):
    """
    k-means on **fixed/reference** PCA tokens only; assign moving by nearest centroid.

    Reference defines the semantic palette (e.g. skull materials). Extra moving
    anatomy (jaw/spine) can only join those cells or land in sparse leftovers —
    it cannot invent new landmark clusters that lack a fixed counterpart.
    """
    from scipy.cluster.vq import kmeans2, vq

    mov_tok = np.asarray(mov_tok, dtype=np.float32)
    fix_tok = np.asarray(fix_tok, dtype=np.float32)
    k = int(k)
    if len(mov_tok) < k or len(fix_tok) < k:
        return None, None, None

    rng = np.random.default_rng(0)
    if len(fix_tok) > int(max_fit):
        fit_idx = rng.choice(len(fix_tok), size=int(max_fit), replace=False)
        fit = fix_tok[fit_idx]
    else:
        fit = fix_tok

    init = fit[rng.choice(len(fit), size=k, replace=False)]
    try:
        centroids, _ = kmeans2(
            fit.astype(np.float64), init.astype(np.float64), iter=25, minit="matrix"
        )
    except Exception:
        return None, None, None
    if centroids is None or len(centroids) < 3:
        return None, None, None
    mov_lab, _ = vq(mov_tok.astype(np.float64), centroids)
    fix_lab, _ = vq(fix_tok.astype(np.float64), centroids)
    return centroids, mov_lab.astype(np.int32), fix_lab.astype(np.int32)


def _fit_joint_semantic_clusters(mov_tok, fix_tok, k, max_fit=MAX_KMEANS_FIT):
    """Deprecated alias — reference-defined clusters (see ``_fit_reference_semantic_clusters``)."""
    return _fit_reference_semantic_clusters(mov_tok, fix_tok, k, max_fit=max_fit)


def _semantic_landmarks_from_clusters(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    k=SEMANTIC_LANDMARK_K,
    min_tokens=None,
):
    """
    One homologous mm landmark per reference-defined PCA-space cluster.

    Centroids are fit on fixed tokens only; moving tokens are assigned to those
    cells. Clusters lacking enough tokens on either side are dropped (extra
    moving anatomy without a fixed counterpart cannot spawn landmarks).

    Returns (packed_tuple, record). packed_tuple is None when landmarks cannot be built.
    """
    record = {"k": int(k), "status": "started"}
    stride = max(1, int(LANDMARK_TOKEN_STRIDE))
    mov_tok, mov_pts = _foreground_token_cloud(
        mov_feat, moving_mask_feat, mov_scales, mov_origin, stride=stride
    )
    fix_tok, fix_pts = _foreground_token_cloud(
        fix_feat, fixed_mask_feat, fix_scales, fix_origin, stride=stride
    )
    record["stride"] = int(stride)
    record["mov_tokens"] = int(len(mov_pts))
    record["fix_tokens"] = int(len(fix_pts))
    if len(mov_pts) < 32 or len(fix_pts) < 32:
        record["status"] = "too_few_tokens"
        return None, record

    if min_tokens is None:
        min_tokens = _adaptive_min_cluster_tokens(k, len(mov_pts), len(fix_pts))
    record["min_cluster_tokens"] = int(min_tokens)

    centroids, mov_lab, fix_lab = _fit_joint_semantic_clusters(mov_tok, fix_tok, k)
    if centroids is None:
        record["status"] = "kmeans_failed"
        return None, record

    mask_m = np.asarray(moving_mask_feat, dtype=bool)
    mask_f = np.asarray(fixed_mask_feat, dtype=bool)
    ii, jj, kk = np.nonzero(mask_m[::stride, ::stride, ::stride])
    mov_ijk = np.stack([ii * stride, jj * stride, kk * stride], axis=1)
    ii, jj, kk = np.nonzero(mask_f[::stride, ::stride, ::stride])
    fix_ijk = np.stack([ii * stride, jj * stride, kk * stride], axis=1)

    mov_desc = _l2_normalize_rows(mov_tok)
    fix_desc = _l2_normalize_rows(fix_tok)
    n_k = int(centroids.shape[0])
    landmarks_m = []
    landmarks_f = []
    scores = []
    cluster_stats = []
    for cid in range(n_k):
        m_sel = mov_lab == cid
        f_sel = fix_lab == cid
        n_m = int(m_sel.sum())
        n_f = int(f_sel.sum())
        if n_m < int(min_tokens) or n_f < int(min_tokens):
            cluster_stats.append({"id": cid, "mov": n_m, "fix": n_f, "kept": False})
            continue
        mean_m = mov_desc[m_sel].mean(axis=0)
        mean_f = fix_desc[f_sel].mean(axis=0)
        # Intrinsic centroid (feature medoid): token with maximal projection onto mean feature
        m_pts_c = mov_pts[m_sel]
        f_pts_c = fix_pts[f_sel]
        c_mov = m_pts_c[int(np.argmax(np.dot(mov_desc[m_sel], mean_m)))]
        c_fix = f_pts_c[int(np.argmax(np.dot(fix_desc[f_sel], mean_f)))]
        nm = float(np.linalg.norm(mean_m))
        nf = float(np.linalg.norm(mean_f))
        cos = float(np.dot(mean_m, mean_f) / max(nm * nf, 1e-8))
        weight = float(np.sqrt(min(n_m, n_f))) * max(cos, 1e-3)
        landmarks_m.append(c_mov)
        landmarks_f.append(c_fix)
        scores.append(weight)
        cluster_stats.append({
            "id": cid, "mov": n_m, "fix": n_f, "kept": True,
            "cosine": cos, "weight": weight,
        })

    record["clusters"] = cluster_stats
    record["n_landmarks"] = len(landmarks_m)
    if len(landmarks_m) < MIN_LANDMARKS:
        record["status"] = "too_few_shared_clusters"
        return None, record

    mov_grid = _scatter_labels_to_grid(mask_m.shape, mov_ijk, mov_lab)
    fix_grid = _scatter_labels_to_grid(mask_f.shape, fix_ijk, fix_lab)
    record["status"] = "ok"
    packed = (
        np.stack(landmarks_m, axis=0),
        np.stack(landmarks_f, axis=0),
        np.asarray(scores, dtype=np.float64),
        mov_grid,
        fix_grid,
        mask_m,
        mask_f,
    )
    return packed, record


def _pool_feature_level(feat_hwzc, mask_hwz, pool):
    """
    Average-pool PCA features and max-pool the foreground mask.

    Returns (pooled_feat, pooled_mask, scale_mult) where scale_mult is the
    factor applied to per-token mm scales.
    """
    p = max(1, int(pool))
    feat = np.asarray(feat_hwzc, dtype=np.float32)
    mask = np.asarray(mask_hwz, dtype=bool)
    if p <= 1:
        return feat, mask, 1.0

    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(np.transpose(feat, (3, 2, 1, 0)))).unsqueeze(0)
    pooled = F.avg_pool3d(t, kernel_size=p, stride=p, ceil_mode=False)
    feat_out = pooled.squeeze(0).permute(3, 2, 1, 0).contiguous().numpy()
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_out = (
        F.max_pool3d(m, kernel_size=p, stride=p, ceil_mode=False).squeeze().numpy() > 0.5
    )
    return feat_out, mask_out, float(p)


def _pyramid_min_mutual_matches(pool):
    p = max(1, int(pool))
    if p >= 4:
        return 6
    if p >= 2:
        return 8
    return MIN_MUTUAL_MATCHES


def _pyramid_match_cosine(pool):
    """Coarse pooled features are blurrier — allow slightly weaker matches."""
    p = max(1, int(pool))
    if p <= 1:
        return MIN_MATCH_COSINE
    return max(0.36, MIN_MATCH_COSINE - 0.05 * float(p - 1))


def _pyramid_min_ransac_inliers(n_matches, pool, nested_fg=False):
    p = max(1, int(pool))
    n = int(n_matches)
    if nested_fg:
        return max(6, min(MIN_RANSAC_INLIERS, n // 5))
    floor = 6 if p >= 4 else 8 if p >= 2 else MIN_RANSAC_INLIERS
    return max(floor, min(MIN_RANSAC_INLIERS, n // 3))


def _pyramid_inlier_mm(mov_diag_mm, fix_diag_mm, pool, nested_fg=False):
    """
    RANSAC inlier radius from foreground scale.

    Nested (small-in-big): tolerance follows the *moving* blob — correspondences
    may be far apart before the rigid, but homologous patches should agree within
    the subject's scale once R,t is found.
    """
    p = max(1, int(pool))
    if nested_fg:
        base = max(5.0, 0.32 * float(mov_diag_mm))
    else:
        base = max(6.0, 0.24 * min(float(mov_diag_mm), float(fix_diag_mm)))
    return base * (1.0 + 0.10 * float(p - 1))


def _mutual_feature_matches(
    moving_desc,
    fixed_desc,
    min_similarity=MIN_MATCH_COSINE,
    max_ratio=LOWE_RATIO,
):
    """Mutual nearest neighbours in cosine space with Lowe ratio (no spatial gate)."""
    if moving_desc.size == 0 or fixed_desc.size == 0:
        return []
    sim = moving_desc @ fixed_desc.T
    n_mov, n_fix = sim.shape

    if n_fix < 2:
        mov_best = np.argmax(sim, axis=1)
        fix_best = np.argmax(sim, axis=0)
        matches = []
        for mi, fi in enumerate(mov_best):
            if not np.isfinite(sim[mi, fi]):
                continue
            if int(fix_best[fi]) != mi:
                continue
            score = float(sim[mi, fi])
            if score < float(min_similarity):
                continue
            matches.append((score, int(mi), int(fi)))
        matches.sort(reverse=True, key=lambda t: t[0])
        return matches

    # Top-2 neighbours without a full sort.
    k2 = np.argpartition(-sim, kth=1, axis=1)[:, :2]
    rows = np.arange(n_mov)
    s0 = sim[rows, k2[:, 0]]
    s1 = sim[rows, k2[:, 1]]
    first_is_0 = s0 >= s1
    mov_best = np.where(first_is_0, k2[:, 0], k2[:, 1])
    second_sim = np.where(first_is_0, s1, s0)
    best_sim = np.where(first_is_0, s0, s1)
    fix_best = np.argmax(sim, axis=0)

    matches = []
    min_sim = float(min_similarity)
    ratio = float(max_ratio)
    for mi, fi in enumerate(mov_best):
        fi = int(fi)
        if not np.isfinite(best_sim[mi]):
            continue
        if int(fix_best[fi]) != mi:
            continue
        score = float(best_sim[mi])
        if score < min_sim:
            continue
        d1 = 1.0 - score
        d2 = 1.0 - float(second_sim[mi])
        if not np.isfinite(d2):
            continue
        if d2 > 1e-6 and d1 > ratio * d2:
            continue
        matches.append((score, int(mi), fi))
    matches.sort(reverse=True, key=lambda t: t[0])
    return matches


def _asymmetric_feature_matches(
    moving_desc,
    fixed_desc,
    *,
    min_similarity=MIN_MATCH_COSINE,
    max_ratio=LOWE_RATIO,
    max_matches=MAX_ASYMMETRIC_MATCHES,
):
    """
    Fixed→moving nearest neighbours (not mutual).

    When token counts differ (large subject, tight reference), mutual NN drops most
    homologous pairs. RANSAC + identity/flip checks disambiguate the extra pairs.
    """
    if moving_desc.size == 0 or fixed_desc.size == 0:
        return []
    sim = moving_desc @ fixed_desc.T
    n_mov, n_fix = sim.shape
    min_sim = float(min_similarity)
    ratio = float(max_ratio)
    matches = []
    if n_mov < 2:
        for fi in range(n_fix):
            mi = int(np.argmax(sim[:, fi]))
            score = float(sim[mi, fi])
            if score >= min_sim:
                matches.append((score, mi, fi))
    else:
        k2 = np.argpartition(-sim, kth=1, axis=0)[:2, :]
        best_mi = k2[0]
        second_mi = k2[1]
        s0 = sim[best_mi, np.arange(n_fix)]
        s1 = sim[second_mi, np.arange(n_fix)]
        swap = s1 > s0
        best_mi = np.where(swap, second_mi, best_mi)
        second_mi = np.where(swap, k2[0], k2[1])
        best_sim = sim[best_mi, np.arange(n_fix)]
        second_sim = sim[second_mi, np.arange(n_fix)]
        for fi in range(n_fix):
            score = float(best_sim[fi])
            if score < min_sim:
                continue
            d1 = 1.0 - score
            d2 = 1.0 - float(second_sim[fi])
            if d2 > 1e-6 and d1 > ratio * d2:
                continue
            matches.append((score, int(best_mi[fi]), fi))
    matches.sort(reverse=True, key=lambda t: t[0])
    if len(matches) <= int(max_matches):
        return matches
    # One moving token → one best fixed partner to avoid collapse.
    kept = []
    used_mov = set()
    for score, mi, fi in matches:
        if mi in used_mov:
            continue
        used_mov.add(mi)
        kept.append((score, mi, fi))
        if len(kept) >= int(max_matches):
            break
    return kept


def _collect_pyramid_correspondences(mov_desc, fix_desc, pool, min_matches, nested_fg=False):
    """
    Semantic correspondences only — no spatial pre-overlap required.

    RANSAC (next step) finds the rigid that best explains homologous pairs.
    Prefer asymmetric matching when the moving blob is nested in a larger reference.
    """
    min_cos = _pyramid_match_cosine(pool)
    mutual_fn = lambda: _mutual_feature_matches(mov_desc, fix_desc, min_similarity=min_cos)
    asym_fn = lambda: _asymmetric_feature_matches(mov_desc, fix_desc, min_similarity=min_cos)
    if nested_fg:
        strategies = (("asymmetric", asym_fn), ("feature_mutual", mutual_fn))
    else:
        strategies = (("feature_mutual", mutual_fn), ("asymmetric", asym_fn))
    best = []
    best_name = "none"
    for name, fn in strategies:
        cand = fn()
        if len(cand) >= int(min_matches):
            return cand, name
        if len(cand) > len(best):
            best = cand
            best_name = name
    return best, best_name


def _nn_feature_matches(src_desc, dst_desc, min_similarity=0.12):
    """Every source token → argmax cosine in dst. No mutual / Lowe gate."""
    if src_desc.size == 0 or dst_desc.size == 0:
        return []
    sim = src_desc @ dst_desc.T
    best = np.argmax(sim, axis=1)
    scores = sim[np.arange(len(src_desc)), best]
    min_sim = float(min_similarity)
    return [
        (float(scores[i]), int(i), int(best[i]))
        for i in range(len(src_desc))
        if np.isfinite(scores[i]) and float(scores[i]) >= min_sim
    ]


def _weighted_kabsch(moving_pts, fixed_pts, weights):
    """Weighted rigid (column) mapping moving → fixed."""
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-8, None)
    mov = np.asarray(moving_pts, dtype=np.float64)
    fix = np.asarray(fixed_pts, dtype=np.float64)
    if mov.shape[0] < 3:
        return None, None
    wsum = float(w.sum())
    c_m = (w[:, None] * mov).sum(axis=0) / wsum
    c_f = (w[:, None] * fix).sum(axis=0) / wsum
    H = (mov - c_m).T @ (w[:, None] * (fix - c_f))
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_f - R @ c_m
    return R, t


def _irls_rigid_on_pairs(moving_pts, fixed_pts, match_scores, R0, t0, scale_mm, n_iter=8):
    """Huber-IRLS so most tokens share one residual valley; outliers (size/part mismatch) drop out."""

    mov = np.asarray(moving_pts, dtype=np.float64)
    fix = np.asarray(fixed_pts, dtype=np.float64)
    scores = np.asarray(match_scores, dtype=np.float64)
    R = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t0, dtype=np.float64).reshape(3)
    s = max(float(scale_mm), 1e-3)
    for _ in range(int(n_iter)):
        res = np.linalg.norm((R @ mov.T).T + t - fix, axis=1)
        w = scores / (1.0 + (res / s) ** 2)
        R_new, t_new = _weighted_kabsch(mov, fix, w)
        if R_new is None:
            break
        R, t = R_new, t_new
    res = np.linalg.norm((R @ mov.T).T + t - fix, axis=1)
    inlier = res <= 2.5 * s
    n_in = int(inlier.sum())
    frac = n_in / max(len(mov), 1)
    med = float(np.median(res[inlier])) if n_in >= 3 else 999.0
    score = float(np.sum(scores[inlier] / (1.0 + res[inlier]))) if n_in >= 3 else -1.0
    return R, t, score, n_in, frac, med


def _rigid_from_semantic_nn_consensus(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    scan_name=None,
    debug=None,
):
    """
    Primary rigid: feature-space NN correspondences, then one consensus R,t.

    Does not require patch-over-patch overlap. Each moving (or fixed) token votes
    for its latent nearest neighbour; IRLS finds the orientation where most votes
    agree spatially.
    """
    from .rigidAlignment import _rigid_rotation_angle_deg, _voxel_patch_ransac_rigid

    pool = 2
    feat_m, mask_m, sm = _pool_feature_level(mov_feat, moving_mask_feat, pool)
    feat_f, mask_f, sf = _pool_feature_level(fix_feat, fixed_mask_feat, pool)
    mov_scales_p = np.asarray(mov_scales, dtype=np.float64) * sm
    fix_scales_p = np.asarray(fix_scales, dtype=np.float64) * sf
    stride = _token_stride(min(int(mask_m.sum()), int(mask_f.sum())))
    mov_tok, mov_pts = _foreground_token_cloud(
        feat_m, mask_m, mov_scales_p, mov_origin, stride=stride
    )
    fix_tok, fix_pts = _foreground_token_cloud(
        feat_f, mask_f, fix_scales_p, fix_origin, stride=stride
    )
    mov_tok, mov_pts = _cap_token_cloud(mov_tok, mov_pts, MAX_MATCH_TOKENS)
    fix_tok, fix_pts = _cap_token_cloud(fix_tok, fix_pts, MAX_MATCH_TOKENS)
    record = {
        "pool": pool,
        "stride": int(stride),
        "mov_tokens": int(len(mov_pts)),
        "fix_tokens": int(len(fix_pts)),
    }
    if len(mov_pts) < 12 or len(fix_pts) < 12:
        record["status"] = "too_few_tokens"
        if debug is not None:
            debug.meta["semantic_nn_consensus"] = record
        return None

    mov_desc = _l2_normalize_rows(mov_tok)
    fix_desc = _l2_normalize_rows(fix_tok)
    clouds = (
        ("mov_to_fix", mov_desc, fix_desc, mov_pts, fix_pts, len(mov_pts)),
        ("fix_to_mov", fix_desc, mov_desc, fix_pts, mov_pts, len(fix_pts)),
    )
    scale_mm = max(3.0, 0.18 * min(float(mov_diag_mm), float(fix_diag_mm)))
    seeds = _orientation_seed_axis_angles()
    best = None
    best_key = (-1.0, -1, -1.0)
    trials = []

    for cloud_name, src_desc, dst_desc, src_pts, dst_pts, n_src in clouds:
        matches = _nn_feature_matches(src_desc, dst_desc, min_similarity=0.12)
        if len(matches) < 12:
            trials.append({"cloud": cloud_name, "n_matches": len(matches), "status": "too_few"})
            continue
        src_corr = np.stack([src_pts[si] for _s, si, _di in matches], axis=0)
        dst_corr = np.stack([dst_pts[di] for _s, _si, di in matches], axis=0)
        scores = np.array([s for s, _si, _di in matches], dtype=np.float64)
        if cloud_name == "fix_to_mov":
            moving_corr, fixed_corr = dst_corr, src_corr
        else:
            moving_corr, fixed_corr = src_corr, dst_corr

        ransac = _voxel_patch_ransac_rigid(
            moving_corr,
            fixed_corr,
            scores,
            n_iterations=500,
            min_inliers=max(8, len(matches) // 8),
            min_spread_mm=max(2.0, 0.06 * min(float(mov_diag_mm), float(fix_diag_mm))),
            inlier_residual_mm=2.5 * scale_mm,
            top_candidates=12,
            scan_name=None,
        )
        seed_list = list(seeds)
        if ransac is not None:
            seed_list = [("ransac", _rotation_to_axis_angle(ransac["R"]))] + seed_list

        cloud_best = None
        for seed_name, aa in seed_list:
            R0 = ransac["R"] if seed_name == "ransac" else _numpy_rodrigues(aa)
            t0 = ransac["t"] if seed_name == "ransac" else (c_f - R0 @ c_m)
            R, t, score, n_in, frac, med = _irls_rigid_on_pairs(
                moving_corr, fixed_corr, scores, R0, t0, scale_mm
            )
            if n_in < 8:
                continue
            key = (frac, n_in, score)
            row = {
                "cloud": cloud_name,
                "seed": seed_name,
                "n_matches": int(len(matches)),
                "n_inliers": int(n_in),
                "inlier_frac": float(frac),
                "score": float(score),
                "median_residual_mm": float(med),
                "angle_deg": float(_rigid_rotation_angle_deg(R)),
            }
            trials.append(row)
            if cloud_best is None or key > best_key:
                cloud_best = (R, t, row)
            if key > best_key:
                best_key = key
                best = (R, t, row)

        if cloud_best is None:
            trials.append({"cloud": cloud_name, "n_matches": len(matches), "status": "no_consensus"})

    record["scale_mm"] = float(scale_mm)
    record["trials"] = trials
    if debug is not None:
        debug.meta["semantic_nn_consensus"] = record

    if best is None:
        record["status"] = "no_consensus"
        _log(scan_name, "semantic-NN consensus: no inlier valley")
        return None

    R, t, row = best
    if float(row["inlier_frac"]) < 0.12:
        record["status"] = "low_inlier_frac"
        _log(
            scan_name,
            f"semantic-NN consensus rejected: inlier_frac={row['inlier_frac']:.3f} "
            f"({row['cloud']} {row['seed']})",
        )
        if debug is not None:
            debug.meta["semantic_nn_consensus"] = record
        return None

    record["status"] = "accepted"
    record["accepted"] = row
    angle = float(row["angle_deg"])
    _log(
        scan_name,
        f"semantic-NN consensus: {row['cloud']} {row['seed']} "
        f"inliers={row['n_inliers']}/{row['n_matches']} "
        f"frac={row['inlier_frac']:.2f} angle={angle:.1f}° med={row['median_residual_mm']:.2f}mm",
    )
    if debug is not None:
        debug.meta["semantic_nn_consensus"] = record
        debug.meta["accepted_pose"] = {
            "R": np.asarray(R, dtype=np.float64).tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "angle_deg": angle,
            "label": "semantic_nn_consensus",
            "source": "semantic_nn_consensus",
            **row,
        }
    return _dino_reg_pose(
        R, t, c_m, c_f, "semantic_nn_consensus",
        angle_deg=angle, inlier_frac=float(row["inlier_frac"]),
    )


def _score_rigid_on_correspondences(R, t, moving_pts, fixed_pts, match_scores, inlier_mm):
    """Feature-weighted inlier score for picking among rotation hypotheses."""
    moving_pts = np.asarray(moving_pts, dtype=np.float64)
    fixed_pts = np.asarray(fixed_pts, dtype=np.float64)
    pred = (np.asarray(R, dtype=np.float64) @ moving_pts.T).T + np.asarray(t, dtype=np.float64).reshape(3)
    residuals = np.linalg.norm(pred - fixed_pts, axis=1)
    inlier_mask = residuals <= float(inlier_mm)
    n_in = int(inlier_mask.sum())
    if n_in < 3:
        return -1.0, n_in, 999.0
    scores = np.asarray(match_scores, dtype=np.float64)[inlier_mask]
    res = residuals[inlier_mask]
    weighted = float(np.sum(scores / (1.0 + res)))
    return weighted, n_in, float(np.median(res))


def _identity_beats_pose(
    weighted,
    n_in,
    med_res,
    id_weighted,
    id_n,
    id_med_res,
    angle_deg,
):
    """
    Return True only when centroid translation is genuinely better than a rotated pose.

    Semantic correspondences often fit identity almost as well as the true rigid
    (similar inlier counts); reject rotation only when identity wins on score or
    the pose has negligible rotation.
    """
    if float(weighted) > float(id_weighted) + 1e-6:
        return False
    if int(n_in) > int(id_n):
        return False
    if float(angle_deg) >= 4.0 and float(weighted) >= float(id_weighted) * 0.99:
        return False
    if float(angle_deg) >= 2.0 and float(med_res) + 0.08 < float(id_med_res):
        return False
    if float(angle_deg) < 2.0 and int(n_in) <= int(id_n) + 1:
        return True
    if float(weighted) < float(id_weighted) * 0.98 and int(n_in) <= int(id_n):
        return True
    return False


def _kabsch_correspondence_fallback(
    moving_corr,
    fixed_corr,
    scores,
    inlier_mm,
    min_inliers,
):
    """When RANSAC finds no consensus, fit Kabsch on high-confidence matches."""
    from .rigidAlignment import _kabsch_rigid, _rigid_rotation_angle_deg

    order = np.argsort(scores)[::-1]
    n = len(scores)
    for frac in (0.45, 0.6, 0.75, 1.0):
        k = max(8, int(n * frac))
        idx = order[:k]
        R, t = _kabsch_rigid(moving_corr[idx], fixed_corr[idx])
        if R is None:
            continue
        weighted, n_in, med_res = _score_rigid_on_correspondences(
            R, t, moving_corr, fixed_corr, scores, inlier_mm
        )
        if n_in < int(min_inliers):
            continue
        return {
            "R": R,
            "t": t,
            "inliers": int(n_in),
            "mean_match": float(np.mean(scores[idx])),
            "angle_deg": float(_rigid_rotation_angle_deg(R)),
            "match_score": float(weighted),
            "median_residual_mm": float(med_res),
        }
    return None


def _rotation_flip_candidates(R, t, c_m, pca_points, base_label="kabsch"):
    """Base rigid plus 180° flips about PCA axes (ALPACA-style disambiguation)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    c_m = np.asarray(c_m, dtype=np.float64)
    candidates = [(str(base_label), R.copy(), t.copy())]

    pts = np.asarray(pca_points, dtype=np.float64)
    if len(pts) >= 6:
        centered = pts - np.mean(pts, axis=0)
        try:
            _, evecs = np.linalg.eigh(np.cov(centered, rowvar=False))
            for axis in evecs[:, ::-1].T:
                flip = _rotation_180_about_axis(axis)
                if flip is None:
                    continue
                R2 = R @ flip
                t2 = R @ (np.eye(3) - flip) @ c_m + t
                candidates.append(("flip180", R2, t2))
        except np.linalg.LinAlgError:
            pass
    return candidates


def _inlier_pairs(moving_pts, fixed_pts, match_scores, R, t, residual_mm):
    """Keep correspondence pairs that agree with y = R x + t within residual_mm."""
    moving_pts = np.asarray(moving_pts, dtype=np.float64)
    fixed_pts = np.asarray(fixed_pts, dtype=np.float64)
    pred = (np.asarray(R, dtype=np.float64) @ moving_pts.T).T + np.asarray(t, dtype=np.float64).reshape(3)
    residuals = np.linalg.norm(pred - fixed_pts, axis=1)
    keep = residuals <= float(residual_mm)
    n_keep = int(keep.sum())
    if n_keep < 3:
        return None
    scores = np.asarray(match_scores, dtype=np.float64)[keep]
    return (
        moving_pts[keep],
        fixed_pts[keep],
        scores,
        float(residuals[keep].mean()),
        n_keep,
    )


def _try_rigid_at_pyramid_level(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    pool,
    scan_name=None,
    debug=None,
):
    """
    One pyramid level: semantic correspondences → RANSAC → flip/identity pick.

    No spatial pre-overlap is required — a smaller subject can align inside a
    larger reference once RANSAC finds the consensus rigid.
    """
    from .rigidAlignment import _rigid_rotation_angle_deg, _voxel_patch_ransac_rigid

    nested_fg = _is_nested_foreground(mov_diag_mm, fix_diag_mm)
    level_record = {
        "pool": int(pool),
        "nested_fg": bool(nested_fg),
        "status": "started",
    }

    feat_p, mask_m_p, sm = _pool_feature_level(mov_feat, moving_mask_feat, pool)
    feat_f, mask_f_p, sf = _pool_feature_level(fix_feat, fixed_mask_feat, pool)
    mov_scales_p = np.asarray(mov_scales, dtype=np.float64) * sm
    fix_scales_p = np.asarray(fix_scales, dtype=np.float64) * sf
    if debug is not None:
        debug.save_pca_semantic_views(
            feat_p, mask_m_p, f"pyramid_div{pool}_moving",
            title_prefix=f"moving ÷{pool}", scales_mm=mov_scales_p,
        )
        debug.save_pca_semantic_views(
            feat_f, mask_f_p, f"pyramid_div{pool}_fixed",
            title_prefix=f"fixed ÷{pool}", scales_mm=fix_scales_p,
        )

    # Same stride on both clouds so mutual NN is not biased toward the smaller volume.
    stride = _token_stride(min(int(mask_m_p.sum()), int(mask_f_p.sum())))
    mov_tok, mov_pts = _foreground_token_cloud(
        feat_p, mask_m_p, mov_scales_p, mov_origin, stride=stride
    )
    fix_tok, fix_pts = _foreground_token_cloud(
        feat_f, mask_f_p, fix_scales_p, fix_origin, stride=stride
    )
    mov_tok, mov_pts = _cap_token_cloud(mov_tok, mov_pts, MAX_MATCH_TOKENS)
    fix_tok, fix_pts = _cap_token_cloud(fix_tok, fix_pts, MAX_MATCH_TOKENS)
    level_record["mov_tokens"] = int(len(mov_pts))
    level_record["fix_tokens"] = int(len(fix_pts))
    level_record["stride_m"] = int(stride)
    level_record["stride_f"] = int(stride)
    if len(mov_pts) < 8 or len(fix_pts) < 8:
        level_record["status"] = "too_few_tokens"
        if debug is not None:
            debug.record_pyramid_level(level_record)
        return None

    mov_desc = _l2_normalize_rows(mov_tok)
    fix_desc = _l2_normalize_rows(fix_tok)
    del mov_tok, fix_tok

    min_matches = _pyramid_min_mutual_matches(pool)
    matches, match_strategy = _collect_pyramid_correspondences(
        mov_desc, fix_desc, pool, min_matches, nested_fg=nested_fg,
    )
    level_record["n_matches"] = int(len(matches))
    level_record["min_matches_required"] = int(min_matches)
    level_record["match_strategy"] = str(match_strategy)
    _log(
        scan_name,
        f"pyramid ÷{pool}: {len(matches)} semantic pairs ({match_strategy}"
        f"{', nested' if nested_fg else ''}) "
        f"(mov={len(mov_pts)} stride={stride}, fix={len(fix_pts)} stride={stride})",
    )
    if debug is not None:
        debug.save_pyramid_match_figure(
            mov_pts,
            fix_pts,
            matches,
            pool,
            0.0,
            status=f"{len(matches)} {match_strategy}",
        )

    if len(matches) < min_matches:
        level_record["status"] = "too_few_matches"
        if debug is not None:
            debug.record_pyramid_level(level_record)
        return None

    moving_corr = np.stack([mov_pts[mi] for _s, mi, _fi in matches], axis=0)
    fixed_corr = np.stack([fix_pts[fi] for _s, _mi, fi in matches], axis=0)
    scores = np.array([s for s, _mi, _fi in matches], dtype=np.float64)

    inlier_mm = _pyramid_inlier_mm(mov_diag_mm, fix_diag_mm, pool, nested_fg=nested_fg)
    spread_mm = max(
        2.0,
        0.06 * (float(mov_diag_mm) if nested_fg else min(float(mov_diag_mm), float(fix_diag_mm))),
    )
    min_inliers = _pyramid_min_ransac_inliers(len(matches), pool, nested_fg=nested_fg)
    level_record["min_inliers_required"] = int(min_inliers)
    level_record["inlier_mm"] = float(inlier_mm)

    refined = _voxel_patch_ransac_rigid(
        moving_corr,
        fixed_corr,
        scores,
        n_iterations=400,
        min_inliers=min_inliers,
        min_spread_mm=spread_mm,
        inlier_residual_mm=inlier_mm,
        top_candidates=12,
        scan_name=None,
    )
    if refined is None:
        loose_mm = inlier_mm * 1.45
        refined = _voxel_patch_ransac_rigid(
            moving_corr,
            fixed_corr,
            scores,
            n_iterations=400,
            min_inliers=max(8, min_inliers - 4),
            min_spread_mm=spread_mm,
            inlier_residual_mm=loose_mm,
            top_candidates=12,
            scan_name=None,
        )
        if refined is not None:
            inlier_mm = loose_mm
            level_record["inlier_mm"] = float(inlier_mm)
    if refined is None:
        fallback_mm = max(inlier_mm * 1.35, 0.28 * min(float(mov_diag_mm), float(fix_diag_mm)))
        refined = _kabsch_correspondence_fallback(
            moving_corr,
            fixed_corr,
            scores,
            fallback_mm,
            max(8, min_inliers - 4),
        )
        if refined is not None:
            inlier_mm = fallback_mm
            level_record["inlier_mm"] = float(inlier_mm)
            level_record["fallback"] = "weighted_kabsch"
    if refined is None:
        _log(
            scan_name,
            f"pyramid ÷{pool}: RANSAC found no rigid with ≥{min_inliers} inliers "
            f"(residual≤{inlier_mm:.1f} mm)",
        )
        level_record["status"] = "ransac_failed"
        level_record["n_inliers"] = 0
        if debug is not None:
            debug.record_pyramid_level(level_record)
        return None

    identity_R = np.eye(3, dtype=np.float64)
    identity_t = np.asarray(c_f, dtype=np.float64) - np.asarray(c_m, dtype=np.float64)
    candidates = _rotation_flip_candidates(
        refined["R"], refined["t"], c_m, fixed_corr, base_label="ransac"
    )
    candidates.append(("identity", identity_R, identity_t))

    best_label = None
    best_state = None
    best_score = (-1.0, -1, 999.0)
    for label, R_c, t_c in candidates:
        weighted, n_in, med_res = _score_rigid_on_correspondences(
            R_c, t_c, moving_corr, fixed_corr, scores, inlier_mm
        )
        pick = (weighted, n_in, -med_res)
        if pick > best_score:
            best_score = pick
            best_label = label
            best_state = (R_c, t_c, weighted, n_in, med_res)

    if best_state is None or best_score[1] < min_inliers:
        level_record["status"] = "hypothesis_below_inlier_threshold"
        level_record["n_inliers"] = int(best_score[1]) if best_score[1] >= 0 else 0
        if debug is not None:
            debug.record_pyramid_level(level_record)
        return None

    R_best, t_best, weighted, n_in, med_res = best_state
    angle = float(_rigid_rotation_angle_deg(R_best))
    level_record["n_inliers"] = int(n_in)
    level_record["best_label"] = str(best_label)
    level_record["score"] = float(weighted)
    level_record["median_residual_mm"] = float(med_res)
    level_record["angle_deg"] = float(angle)
    id_weighted, id_n, id_med = _score_rigid_on_correspondences(
        identity_R, identity_t, moving_corr, fixed_corr, scores, inlier_mm
    )
    level_record["identity_score"] = float(id_weighted)
    level_record["identity_inliers"] = int(id_n)
    level_record["identity_median_residual_mm"] = float(id_med)
    if best_label != "identity" and _identity_beats_pose(
        weighted, n_in, med_res, id_weighted, id_n, id_med, angle
    ):
        _log(
            scan_name,
            f"pyramid ÷{pool}: keeping centroid pre-align "
            f"(pose angle={angle:.1f}° score={weighted:.1f} vs identity={id_weighted:.1f})",
        )
        level_record["status"] = "identity_preferred"
        if debug is not None:
            debug.record_pyramid_level(level_record)
        return None

    level_record["status"] = "accepted_candidate"
    if debug is not None:
        debug.record_pyramid_level(level_record)
    _log(
        scan_name,
        f"pyramid ÷{pool} candidate ({best_label}, {match_strategy}): angle={angle:.1f}° "
        f"inliers={n_in}/{len(matches)} med_res={med_res:.2f} mm "
        f"score={weighted:.1f} (identity={id_weighted:.1f})",
    )
    # Prefer RANSAC/inlier pairs for landmark viz; fall back to all mutual matches.
    corr = {
        "mov_pts": np.asarray(moving_corr, dtype=np.float64),
        "fix_pts": np.asarray(fixed_corr, dtype=np.float64),
        "scores": np.asarray(scores, dtype=np.float64),
        "inlier_mm": float(inlier_mm),
        "match_strategy": str(match_strategy),
    }
    inl = _inlier_pairs(
        moving_corr, fixed_corr, scores, R_best, t_best, inlier_mm
    )
    if inl is not None:
        corr["mov_pts"] = np.asarray(inl[0], dtype=np.float64)
        corr["fix_pts"] = np.asarray(inl[1], dtype=np.float64)
        corr["scores"] = np.asarray(inl[2], dtype=np.float64)
        corr["used_inliers"] = True
    else:
        corr["used_inliers"] = False
    return R_best, t_best, best_label, (weighted, n_in, pool, angle), corr


def _pick_rigid_from_landmark_pairs(
    moving_corr,
    fixed_corr,
    scores,
    c_m,
    c_f,
    inlier_mm,
    spread_mm,
    min_inliers,
    scan_name=None,
    stage_label="",
    max_angle_deg=None,
):
    """
    Homologous cluster CoMs → weighted Kabsch (+ 180° flip / identity pick).

    Cluster ids already define correspondences, so RANSAC is the wrong tool here
    (RANSAC is for unknown/noisy NN matches). Pure weighted Kabsch is primary;
    ``spread_mm`` is unused but kept for call-site compatibility.
    ``max_angle_deg`` constrains accepted candidates to small residuals when pre-aligned.
    """
    from .rigidAlignment import _rigid_rotation_angle_deg

    del spread_mm  # landmarks are already homologous; no RANSAC spread gate
    moving_corr = np.asarray(moving_corr, dtype=np.float64)
    fixed_corr = np.asarray(fixed_corr, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(moving_corr) < 3:
        return None

    R0, t0 = _weighted_kabsch(moving_corr, fixed_corr, scores)
    if R0 is None:
        return None
    w0, n0, med0 = _score_rigid_on_correspondences(
        R0, t0, moving_corr, fixed_corr, scores, inlier_mm
    )
    if n0 < max(3, int(min_inliers) - 1):
        return None
    refined = {
        "R": R0,
        "t": t0,
        "inliers": int(n0),
        "angle_deg": float(_rigid_rotation_angle_deg(R0)),
        "match_score": float(w0),
        "median_residual_mm": float(med0),
        "estimator": "weighted_kabsch",
    }
    if scan_name and stage_label:
        _log(
            scan_name,
            f"{stage_label.strip()}: Kabsch angle={refined['angle_deg']:.1f}° "
            f"inliers={n0}/{len(moving_corr)} med_res={med0:.2f} mm",
        )

    identity_R = np.eye(3, dtype=np.float64)
    identity_t = np.asarray(c_f, dtype=np.float64) - np.asarray(c_m, dtype=np.float64)
    candidates = _rotation_flip_candidates(
        refined["R"], refined["t"], c_m, fixed_corr, base_label="kabsch"
    )
    candidates.append(("identity", identity_R, identity_t))

    best_label = None
    best_state = None
    best_score = (-1.0, -1, 999.0)
    for label, R_c, t_c in candidates:
        ang = float(_rigid_rotation_angle_deg(R_c))
        if max_angle_deg is not None and ang > float(max_angle_deg):
            continue
        weighted, n_in, med_res = _score_rigid_on_correspondences(
            R_c, t_c, moving_corr, fixed_corr, scores, inlier_mm
        )
        pick = (weighted, n_in, -med_res)
        if pick > best_score:
            best_score = pick
            best_label = label
            best_state = (R_c, t_c, weighted, n_in, med_res)

    if best_state is None or best_score[1] < 3:
        return None

    R_best, t_best, weighted, n_in, med_res = best_state
    angle = float(_rigid_rotation_angle_deg(R_best))
    id_weighted, id_n, id_med = _score_rigid_on_correspondences(
        identity_R, identity_t, moving_corr, fixed_corr, scores, inlier_mm
    )
    if (
        max_angle_deg is None
        and best_label != "identity"
        and _identity_beats_pose(
            weighted, n_in, med_res, id_weighted, id_n, id_med, angle
        )
    ):
        return None

    return {
        "R": R_best,
        "t": t_best,
        "label": str(best_label),
        "angle_deg": float(angle),
        "n_inliers": int(n_in),
        "score": float(weighted),
        "median_residual_mm": float(med_res),
        "identity_score": float(id_weighted),
        "identity_inliers": int(id_n),
        "estimator": "weighted_kabsch",
    }


def _dino_reg_pose(R_col, t_mm, scan_centroid_mm, reference_centroid_mm, source, **extra):
    """Physical rigid pose for direct scipy volume warp (no proxy guidepoints)."""
    return {
        "R": np.asarray(R_col, dtype=np.float64).reshape(3, 3),
        "t": np.asarray(t_mm, dtype=np.float64).reshape(3),
        "scan_centroid_mm": np.asarray(scan_centroid_mm, dtype=np.float64).reshape(3),
        "reference_centroid_mm": np.asarray(reference_centroid_mm, dtype=np.float64).reshape(3),
        "source": str(source),
        **extra,
    }


def _pose_refined_cosine(pose):
    """Finite refined cosine, or NaN if missing/invalid."""
    if not pose:
        return float("nan")
    try:
        v = float(pose.get("refined_cosine", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _pose_cosine_rank_key(pose):
    v = _pose_refined_cosine(pose)
    return v if np.isfinite(v) else -999.0


def _numpy_rodrigues(axis_angle):
    """Column-vector Rodrigues: p' = R @ p."""
    w = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rotation_to_axis_angle(R):
    """Log map of a proper rotation to axis-angle (radians)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    cos_t = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        # 180°: pick the largest diagonal
        axis = np.sqrt(np.maximum(0.0, (np.diag(R) + 1.0) * 0.5))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[1] = np.copysign(axis[1], R[0, 1])
            axis[2] = np.copysign(axis[2], R[0, 2])
        elif axis[1] >= axis[2]:
            axis[0] = np.copysign(axis[0], R[0, 1])
            axis[2] = np.copysign(axis[2], R[1, 2])
        else:
            axis[0] = np.copysign(axis[0], R[0, 2])
            axis[1] = np.copysign(axis[1], R[1, 2])
        norm = float(np.linalg.norm(axis))
    return (axis / max(norm, 1e-12)) * theta


def _octahedral_rotations():
    """24 proper 90° axis permutations (includes the typical tongue re-lay)."""
    import itertools

    out = []
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            for col, (axis, sgn) in enumerate(zip(perm, signs)):
                R[int(axis), col] = float(sgn)
            if abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
                continue
            name = "".join("xyz"[p] + ("+" if s > 0 else "-") for p, s in zip(perm, signs))
            out.append((name, R))
    return tuple(out)


def _centroid_locked_t(R, c_m, c_f):
    """t for p_fixed = R @ p_moving + t when centroids map to each other."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    return c_f - R @ c_m


def _foreground_bbox_slices(mask, pad=2):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return tuple(slice(0, int(s)) for s in mask.shape)
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - int(pad), 0)
    hi = np.minimum(coords.max(axis=0) + int(pad) + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _fg_crop_to_padded_cube(volume, mask, cube=OCTAHEDRAL_CANVAS):
    """
    Isotropic thumbnail of the FG: preserve aspect, pad to ``cube³``.

    Independent per-axis squash into a cube would destroy the layout octahedral
    search is trying to match.
    """
    volume = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    sl = _foreground_bbox_slices(mask, pad=1)
    crop_v = volume[sl]
    crop_m = mask[sl]
    shape = np.array(crop_v.shape, dtype=np.float64)
    cube = int(cube)
    inner = max(int(cube) - 2, 8)
    scale = float(inner) / max(float(shape.max()), 1.0)
    new_shape = np.maximum(1, np.round(shape * scale).astype(np.int32))
    resized_v = resize(
        crop_v,
        tuple(int(s) for s in new_shape),
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)
    resized_m = (
        resize(
            crop_m.astype(np.float32),
            tuple(int(s) for s in new_shape),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )
        > 0.5
    )
    out_v = np.zeros((cube, cube, cube), dtype=np.float32)
    out_m = np.zeros((cube, cube, cube), dtype=bool)
    start = (np.array([cube, cube, cube]) - new_shape) // 2
    end = start + new_shape
    out_v[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = resized_v
    out_m[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = resized_m
    return out_v, out_m


def _apply_octahedral_relabel(vol, R):
    """
    Exact 90° axis permute + sign flip: ``out[q] = in[R.T @ (q-c) + c]``.

    Octahedral poses map the voxel grid onto itself, so this is a transpose/flip
    — no interpolation, no mirrors (caller must pass det(R)=+1).
    """
    vol = np.asarray(vol)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    src_axes = []
    flips = []
    for out_ax in range(3):
        in_ax = int(np.argmax(np.abs(R[out_ax])))
        src_axes.append(in_ax)
        flips.append(float(R[out_ax, in_ax]) < 0.0)
    out = np.transpose(vol, axes=src_axes)
    for out_ax, do_flip in enumerate(flips):
        if do_flip:
            out = np.flip(out, axis=out_ax)
    return np.ascontiguousarray(out)


def _rotate_volume_about_center(vol, R, order=1):
    """Interpolating rotate about array center. Prefer ``_apply_octahedral_relabel`` for 90° cubes."""
    vol = np.asarray(vol)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = (np.array(vol.shape, dtype=np.float64) - 1.0) * 0.5
    M = R.T
    offset = c - M @ c
    return ndimage.affine_transform(
        vol,
        M,
        offset=offset,
        order=int(order),
        prefilter=order > 1,
    )


def _cube_slice_indices(mask_cube, n_slices):
    """Even Z samples through the cube FG (fallback: whole cube)."""
    mask = np.asarray(mask_cube, dtype=bool)
    n_slices = max(3, int(n_slices))
    z_n = int(mask.shape[2])
    proj = mask.any(axis=(0, 1))
    if proj.any():
        z0 = int(np.flatnonzero(proj)[0])
        z1 = int(np.flatnonzero(proj)[-1])
    else:
        z0, z1 = 0, z_n - 1
    if z1 <= z0:
        return np.array([min(z0, z_n - 1)], dtype=np.int32)
    idx = np.linspace(z0, z1, n_slices)
    return np.unique(np.clip(np.round(idx).astype(np.int32), 0, z_n - 1))


def _encode_cube_axial_tokens(
    volume_cube,
    mask_cube,
    model,
    device,
    feat_h,
    feat_w,
    z_index,
    batch_size,
    scan_name=None,
    label="oct-ax",
):
    """
    Axial DINO on selected cube slices → L2-normalized tokens (S, HW, C) + FG mask.
    """
    import torch

    volume_cube = np.asarray(volume_cube, dtype=np.float32)
    mask_cube = np.asarray(mask_cube, dtype=bool)
    z_index = np.asarray(z_index, dtype=np.int32)
    input_h = int(feat_h) * PATCH_SIZE
    input_w = int(feat_w) * PATCH_SIZE
    hw = int(feat_h) * int(feat_w)
    embed_dim = int(getattr(model, "embed_dim", 1024))
    n = int(len(z_index))
    tokens = np.zeros((n, hw, embed_dim), dtype=np.float32)
    tok_mask = np.zeros((n, hw), dtype=bool)
    bs = max(1, int(batch_size))
    i0 = 0
    while i0 < n:
        batch_ids = list(range(i0, min(i0 + bs, n)))
        sl = np.stack([volume_cube[:, :, int(z_index[j])] for j in batch_ids], axis=0)
        try:
            toks = _encode_slice_batch_gpu(model, device, sl, input_h, input_w)
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                new_bs = _shrink_encode_batch(bs, scan_name=scan_name, label=label)
                if new_bs is None:
                    raise
                bs = new_bs
                continue
            raise
        toks = _l2_normalize_tokens_bnC(toks)
        tokens[batch_ids] = toks.detach().cpu().numpy().astype(np.float32)
        del toks
        for j in batch_ids:
            sm = resize(
                mask_cube[:, :, int(z_index[j])].astype(np.float32),
                (feat_h, feat_w),
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            ) > 0.5
            tok_mask[j] = sm.reshape(-1)
        i0 += len(batch_ids)
    return tokens, tok_mask


def _masked_token_cosine(mov_tok, mov_mask, fix_tok, fix_mask):
    overlap = np.asarray(mov_mask, dtype=bool) & np.asarray(fix_mask, dtype=bool)
    n = int(overlap.sum())
    if n < 8:
        return float("nan")
    a = np.asarray(mov_tok, dtype=np.float32)[overlap]
    b = np.asarray(fix_tok, dtype=np.float32)[overlap]
    return float((a * b).sum(axis=-1).mean())


def _octahedral_cheap_encode_search(
    moving_volume,
    fixed_volume,
    moving_mask,
    fixed_mask,
    c_m,
    c_f,
    model,
    device,
    scan_name=None,
    debug=None,
):
    """
    Pass 1: pose the moving thumbnail with each of 24 proper octahedral rotations,
    axial-encode, masked cosine vs the fixed thumbnail (encoded once).

    Returns dict with R, t (centroid-locked mm), scores, and ``weak`` when the
    top cubes are too close or the best cosine is too low.
    """
    from .rigidAlignment import _rigid_rotation_angle_deg

    feat_h, feat_w = OCTAHEDRAL_FEAT_GRID
    mov_c, mov_m = _fg_crop_to_padded_cube(moving_volume, moving_mask, OCTAHEDRAL_CANVAS)
    fix_c, fix_m = _fg_crop_to_padded_cube(fixed_volume, fixed_mask, OCTAHEDRAL_CANVAS)
    z_index = _cube_slice_indices(fix_m, OCTAHEDRAL_SLICES)
    _log(
        scan_name,
        f"octahedral pass-1 canvas={OCTAHEDRAL_CANVAS}³ slices={len(z_index)} "
        f"grid={feat_h}x{feat_w} (axial DINO after each 90° relabel)",
    )
    sample = fix_c[:, :, int(z_index[len(z_index) // 2])]
    cache = {}
    bs = _cached_encode_batch_size(
        model,
        device,
        feat_h,
        feat_w,
        sample,
        cache,
        max_bs=min(int(ENCODE_BATCH_MAX), max(8, len(z_index))),
        scan_name=scan_name,
        label="oct-probe",
    )
    bs = max(int(bs), 1)
    fix_tok, fix_tok_m = _encode_cube_axial_tokens(
        fix_c, fix_m, model, device, feat_h, feat_w, z_index,
        batch_size=bs, scan_name=scan_name, label="oct-fixed",
    )

    probes = []
    best = None
    best_cos = -999.0
    second_cos = -999.0
    for name, R in _octahedral_rotations():
        posed_v = _apply_octahedral_relabel(mov_c, R)
        posed_m = _apply_octahedral_relabel(mov_m, R)
        mov_tok, mov_tok_m = _encode_cube_axial_tokens(
            posed_v, posed_m, model, device, feat_h, feat_w, z_index,
            batch_size=bs, scan_name=scan_name, label=f"oct-{name}",
        )
        cos = _masked_token_cosine(mov_tok, mov_tok_m, fix_tok, fix_tok_m)
        angle = float(_rigid_rotation_angle_deg(R))
        row = {
            "name": str(name),
            "cosine": float(cos) if np.isfinite(cos) else float("nan"),
            "angle_deg": angle,
        }
        probes.append(row)
        val = float(cos) if np.isfinite(cos) else -999.0
        if val > best_cos:
            second_cos = best_cos
            best_cos = val
            best = (name, R, val, angle)
        elif val > second_cos:
            second_cos = val
        _log(
            scan_name,
            f"octahedral {name}: cos={cos:.3f} angle={angle:.1f}°",
        )

    if best is None:
        return None

    name, R, cos0, angle0 = best
    margin = float(best_cos - second_cos) if second_cos > -900 else float(best_cos)
    weak = bool(
        (not np.isfinite(cos0))
        or float(cos0) < float(OCTAHEDRAL_MIN_COSINE)
        or margin < float(OCTAHEDRAL_SCORE_MARGIN)
    )
    t = _centroid_locked_t(R, c_m, c_f)
    ranked = sorted(
        probes,
        key=lambda p: p["cosine"] if np.isfinite(p["cosine"]) else -999.0,
        reverse=True,
    )
    record = {
        "status": "ok",
        "canvas": int(OCTAHEDRAL_CANVAS),
        "slices": int(len(z_index)),
        "feat_grid": [int(feat_h), int(feat_w)],
        "best_name": str(name),
        "best_cosine": float(cos0),
        "second_cosine": float(second_cos) if second_cos > -900 else None,
        "margin": float(margin),
        "weak": weak,
        "angle_deg": float(angle0),
        "probes": ranked,
    }
    _log(
        scan_name,
        f"octahedral pass-1 winner {name} cos={cos0:.3f} "
        f"margin={margin:.3f} weak={weak} angle={angle0:.1f}°",
    )
    if debug is not None:
        debug.meta["octahedral_pass1"] = record
        try:
            path = os.path.join(debug.debug_dir, "octahedral_pass1.txt")
            lines = [
                f"winner={name}  cos={cos0:.4f}  margin={margin:.4f}  weak={weak}",
                "",
            ]
            for p in ranked:
                mark = " <-- best" if p["name"] == name else ""
                lines.append(
                    f"{p['name']:12s}  cos={p['cosine']:.4f}  "
                    f"angle={p['angle_deg']:.1f}°{mark}"
                )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception:
            pass
    return {
        "R": np.asarray(R, dtype=np.float64),
        "t": np.asarray(t, dtype=np.float64),
        "name": str(name),
        "cosine": float(cos0),
        "margin": float(margin),
        "weak": weak,
        "angle_deg": float(angle0),
        "record": record,
    }


def _score_feature_cosine_at_R(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    R,
    c_m,
    c_f,
    pool=SEMANTIC_K_RANK_POOL,
):
    """Masked 24-D feature cosine at a centroid-locked rotation (high-res grid)."""
    import torch

    device = torch.device("cuda", 0)
    tensors = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        pool,
        device,
    )
    if tensors is None:
        return float("nan")
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    delta = np.zeros(3, dtype=np.float32)
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)
    with torch.no_grad():
        cos_t = _feature_map_cosine_given_R(
            tensors["mov_vol"],
            tensors["fix_vol"],
            tensors["fix_mask"],
            tensors["mov_mask"],
            tensors["fixed_mm"],
            tensors["mov_origin_t"],
            tensors["mov_scales_t"],
            tensors["mov_shape"],
            R,
            c_m_t,
            c_f_t,
            torch.as_tensor(delta, dtype=torch.float32, device=device),
        )
        cos = float(cos_t.detach().cpu())
    del tensors
    torch.cuda.empty_cache()
    return cos


def _encode_pair_highest_res(
    moving_crop,
    fixed_crop,
    moving_mask_c,
    fixed_mask_c,
    model,
    device,
    scan_name=None,
    progress_callback=None,
):
    """Pass 2: stream tri-planar joint PCA, largest feature grid that fits."""
    last_oom = None
    feat_h = feat_w = None
    mov_feat = fix_feat = None
    for feat_h, feat_w in FEAT_GRID_CANDIDATES:
        _log(
            scan_name,
            f"pass-2 tri-planar {feat_h}x{feat_w} gap={SLICE_GAP}, "
            f"moving={moving_crop.shape} fixed={fixed_crop.shape}",
        )
        _notify(
            progress_callback,
            f"Encoding DINOv3 tri-planar ({feat_h}x{feat_w}) for {scan_name or 'scan'}...",
        )
        try:
            mov_feat, fix_feat = encode_pair_triplanar_joint_pca(
                moving_crop,
                fixed_crop,
                model,
                device,
                feat_h,
                feat_w,
                moving_mask=moving_mask_c,
                fixed_mask=fixed_mask_c,
                gap=SLICE_GAP,
                n_components=REG_FEATURE_DIM,
                scan_name=scan_name,
                batch_size=ENCODE_BATCH_SIZE,
            )
            last_oom = None
            break
        except (RuntimeError, MemoryError) as exc:
            msg = str(exc).lower()
            if isinstance(exc, MemoryError) or "out of memory" in msg:
                last_oom = exc
                _log(scan_name, f"CUDA/host OOM at {feat_h}x{feat_w}, trying a smaller grid")
                import gc
                import torch

                mov_feat = fix_feat = None
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise DinoRegError(f"DINOv3 feature extraction failed: {exc}") from exc
    else:
        raise DinoRegError(
            "Out of memory during DINO-Reg tri-planar encoding; "
            "close other GPU/RAM jobs or use manual guidepoints."
        ) from last_oom
    return mov_feat, fix_feat, feat_h, feat_w


def _orientation_seed_axis_angles():
    """
    Identity, ±90° / 180° about each axis, and two-axis 90° compositions.

    Rotations are about the moving centroid (see ``_feature_map_cosine_at_pose``),
    so these seeds keep the blobs overlapping and only change orientation.
    """
    half = 0.5 * np.pi
    axes = (
        ("rx", np.array([1.0, 0.0, 0.0])),
        ("ry", np.array([0.0, 1.0, 0.0])),
        ("rz", np.array([0.0, 0.0, 1.0])),
    )
    seeds = [("identity", np.zeros(3, dtype=np.float64))]
    for name, ax in axes:
        seeds.append((f"{name}+90", ax * half))
        seeds.append((f"{name}-90", -ax * half))
        seeds.append((f"{name}+180", ax * np.pi))
    for i, (n1, a1) in enumerate(axes):
        for n2, a2 in axes[i + 1 :]:
            for s1, l1 in ((1.0, "+90"), (-1.0, "-90")):
                for s2, l2 in ((1.0, "+90"), (-1.0, "-90")):
                    R = _numpy_rodrigues(s2 * half * a2) @ _numpy_rodrigues(s1 * half * a1)
                    seeds.append((f"{n1}{l1}_{n2}{l2}", _rotation_to_axis_angle(R)))
    return tuple((name, tuple(float(x) for x in aa)) for name, aa in seeds)


def _axis_angle_to_rotation_matrix(axis_angle):
    """
    Differentiable Rodrigues (column action p' = R @ p).

    Stable at the origin so identity seeds still receive rotation gradients.
    """
    import torch

    w = torch.as_tensor(axis_angle, dtype=torch.float32).reshape(3)
    wx, wy, wz = w[0], w[1], w[2]
    zero = w.new_zeros(())
    K = torch.stack(
        [
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ],
        dim=0,
    )
    theta2 = (w * w).sum()
    theta = torch.sqrt(theta2 + 1e-12)
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12)
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    return eye + A * K + B * (K @ K)


def _feat_hwzd_to_torch(feat_hwzd):
    """Numpy (H, W, D, C) foreground PCA grid → torch (1, C, D, H, W)."""
    import torch

    feat = np.ascontiguousarray(feat_hwzd, dtype=np.float32)
    t = torch.from_numpy(np.transpose(feat, (3, 2, 0, 1)))
    return t.unsqueeze(0)


def _fixed_mm_grid(origin_mm, scales_mm, shape_hwz, device):
    """
    Per-voxel physical mm centres on the fixed feature grid.

    Returns (D, H, W, 3) with last axis ``[x_mm, y_mm, z_mm]`` (same frame as
    ``_foreground_token_cloud``).  ``_mm_to_mov_grid_norm`` maps these to
    ``grid_sample`` normalized coords.
    """
    import torch

    H, W, D = (int(shape_hwz[0]), int(shape_hwz[1]), int(shape_hwz[2]))
    origin = torch.as_tensor(origin_mm, dtype=torch.float32, device=device).reshape(3)
    scales = torch.as_tensor(scales_mm, dtype=torch.float32, device=device).reshape(3)
    ii = torch.arange(H, device=device, dtype=torch.float32)
    jj = torch.arange(W, device=device, dtype=torch.float32)
    kk = torch.arange(D, device=device, dtype=torch.float32)
    i_grid, j_grid, k_grid = torch.meshgrid(ii, jj, kk, indexing="ij")
    x_mm = origin[0] + (i_grid + 0.5) * scales[0]
    y_mm = origin[1] + (j_grid + 0.5) * scales[1]
    z_mm = origin[2] + (k_grid + 0.5) * scales[2]
    return torch.stack([x_mm, y_mm, z_mm], dim=-1).permute(2, 0, 1, 3)


def _mm_to_mov_grid_norm(p_mm, origin_mm, scales_mm, shape_hwz):
    """
    Map mm points to normalized moving-grid coords for ``grid_sample``.

    ``p_mm`` has trailing shape (..., 3) with (x, y, z) mm; returns (..., 3) in [-1, 1].
    """
    import torch

    H, W, D = (int(shape_hwz[0]), int(shape_hwz[1]), int(shape_hwz[2]))
    origin = torch.as_tensor(origin_mm, dtype=p_mm.dtype, device=p_mm.device).reshape(3)
    scales = torch.as_tensor(scales_mm, dtype=p_mm.dtype, device=p_mm.device).reshape(3)
    ix = (p_mm[..., 0] - origin[0]) / scales[0] - 0.5
    iy = (p_mm[..., 1] - origin[1]) / scales[1] - 0.5
    iz = (p_mm[..., 2] - origin[2]) / scales[2] - 0.5
    nx = 2.0 * ((iy + 0.5) / float(W)) - 1.0
    ny = 2.0 * ((ix + 0.5) / float(H)) - 1.0
    nz = 2.0 * ((iz + 0.5) / float(D)) - 1.0
    return torch.stack([nx, ny, nz], dim=-1)


def _mask_hwz_to_torch(mask_hwz, device):
    """Numpy (H, W, D) foreground mask → torch (1, 1, D, H, W) matching feature volumes."""
    import torch

    m = torch.from_numpy(np.asarray(mask_hwz, dtype=np.float32)).to(device)
    return m.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


def _feature_map_cosine_given_R(
    mov_vol,
    fix_vol,
    fix_mask,
    mov_mask,
    fixed_mm_grid,
    mov_origin,
    mov_scales,
    mov_shape_hwz,
    R_col,
    c_m,
    c_f,
    delta,
):
    """Centroid-locked feature cosine for a fixed rotation matrix (no IRLS)."""
    import torch
    import torch.nn.functional as F

    R = torch.as_tensor(R_col, dtype=torch.float32, device=fixed_mm_grid.device)
    p_m = (fixed_mm_grid - c_f.view(1, 1, 1, 3) - delta.view(1, 1, 1, 3)) @ R + c_m.view(1, 1, 1, 3)
    grid = _mm_to_mov_grid_norm(p_m, mov_origin, mov_scales, mov_shape_hwz).unsqueeze(0)
    warped = F.grid_sample(
        mov_vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_mask = F.grid_sample(
        mov_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_n = F.normalize(warped, dim=1, eps=1e-6)
    fix_n = F.normalize(fix_vol, dim=1, eps=1e-6)
    cos = (warped_n * fix_n).sum(dim=1, keepdim=True)
    overlap = fix_mask * warped_mask
    denom = overlap.sum().clamp_min(1.0)
    return (cos * overlap).sum() / denom


def _semantic_dice_cosine_eval(
    mov_vol,
    fix_vol,
    fix_mask,
    mov_mask,
    fixed_mm_grid,
    mov_origin,
    mov_scales,
    mov_shape_hwz,
    R_or_axis_angle,
    c_m_t,
    c_f_t,
    delta_t=None,
):
    """
    Evaluates volumetric foreground Dice and semantic feature Cosine.

    Score = cosine * min(1, dice / OCTAHEDRAL_MIN_OVERLAP_DICE) + DINO_REG_DICE_WEIGHT * dice
    so tissue correspondence dominates while low-overlap poses stay gated.

    Forward: ``p_fixed = R @ (p_moving - c_m) + c_f + delta``.
    Inverse sampling for ``grid_sample``:
    ``p_moving = R.T @ (p_fixed - c_f - delta) + c_m``.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(R_or_axis_angle, torch.Tensor) and R_or_axis_angle.ndim == 1:
        R = _axis_angle_to_rotation_matrix(R_or_axis_angle)
    else:
        R = torch.as_tensor(R_or_axis_angle, dtype=torch.float32, device=fixed_mm_grid.device)

    if delta_t is not None:
        p_m = (fixed_mm_grid - c_f_t.view(1, 1, 1, 3) - delta_t.view(1, 1, 1, 3)) @ R + c_m_t.view(1, 1, 1, 3)
    else:
        p_m = (fixed_mm_grid - c_f_t.view(1, 1, 1, 3)) @ R + c_m_t.view(1, 1, 1, 3)

    grid = _mm_to_mov_grid_norm(p_m, mov_origin, mov_scales, mov_shape_hwz).unsqueeze(0)

    warped_mask = F.grid_sample(
        mov_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    inter = (fix_mask * warped_mask).sum()
    dice = (2.0 * inter) / (fix_mask.sum() + warped_mask.sum()).clamp_min(1.0)
    overlap = fix_mask * warped_mask

    warped_vol = F.grid_sample(
        mov_vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_n = F.normalize(warped_vol, dim=1, eps=1e-6)
    fix_n = F.normalize(fix_vol, dim=1, eps=1e-6)
    cos_map = (warped_n * fix_n).sum(dim=1, keepdim=True)
    cosine = (cos_map * overlap).sum() / overlap.sum().clamp_min(1.0)

    # Semantic correspondence alignment is primary, stabilized by overlap:
    gate = (dice / float(OCTAHEDRAL_MIN_OVERLAP_DICE)).clamp(max=1.0)
    score = cosine * gate + float(DINO_REG_DICE_WEIGHT) * dice
    return score, dice, cosine


def _adam_refine_hybrid_pose(
    tensors,
    *,
    init_axis_angle,
    c_m,
    c_f,
    init_delta,
    n_steps,
    lr,
):
    """Adam on axis-angle + residual translation maximizing semantic score."""
    import torch

    device = tensors["mov_vol"].device
    axis_angle = torch.as_tensor(init_axis_angle, dtype=torch.float32, device=device).clone()
    axis_angle.requires_grad_(True)
    delta = torch.as_tensor(init_delta, dtype=torch.float32, device=device).clone()
    delta.requires_grad_(True)
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([axis_angle, delta], lr=float(lr))

    def _eval(aa, dlt):
        return _semantic_dice_cosine_eval(
            tensors["mov_vol"],
            tensors["fix_vol"],
            tensors["fix_mask"],
            tensors["mov_mask"],
            tensors["fixed_mm"],
            tensors["mov_origin_t"],
            tensors["mov_scales_t"],
            tensors["mov_shape"],
            aa,
            c_m_t,
            c_f_t,
            dlt,
        )

    with torch.no_grad():
        init_score, init_dice, init_cos = _eval(axis_angle.detach(), delta.detach())
        best = {
            "score": float(init_score.cpu()),
            "dice": float(init_dice.cpu()),
            "cosine": float(init_cos.cpu()),
            "init_score": float(init_score.cpu()),
            "init_dice": float(init_dice.cpu()),
            "init_cosine": float(init_cos.cpu()),
            "axis_angle": axis_angle.detach().clone(),
            "delta": delta.detach().clone(),
        }

    with torch.enable_grad():
        for _step in range(int(n_steps)):
            score, dice, cosine = _eval(axis_angle, delta)
            loss = -score
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            score_val = float(score.detach().cpu())
            if score_val > best["score"]:
                best["score"] = score_val
                best["dice"] = float(dice.detach().cpu())
                best["cosine"] = float(cosine.detach().cpu())
                best["axis_angle"] = axis_angle.detach().clone()
                best["delta"] = delta.detach().clone()
    return best


def _masked_intensity_ncc(warped_int, fix_int, overlap, eps=1e-6):
    """Scalar NCC over the overlap mask (single-channel volumes)."""
    import torch

    w = overlap.clamp(0.0, 1.0)
    denom = w.sum().clamp_min(1.0)
    m_w = (warped_int * w).sum() / denom
    m_f = (fix_int * w).sum() / denom
    dw = (warped_int - m_w) * w
    df = (fix_int - m_f) * w
    num = (dw * df).sum()
    den = torch.sqrt((dw * dw).sum() * (df * df).sum()).clamp_min(eps)
    return num / den


def _semantic_cosine_intensity_eval(
    mov_vol,
    fix_vol,
    fix_mask,
    mov_mask,
    fixed_mm_grid,
    mov_origin,
    mov_scales,
    mov_shape_hwz,
    axis_angle,
    c_m,
    c_f,
    delta,
    mov_int=None,
    fix_int=None,
    intensity_weight=SEMANTIC_REFINE_INTENSITY_WEIGHT,
):
    """
    Masked mean feature cosine (+ optional weak intensity NCC). No Dice.

    Score = cosine + intensity_weight * ncc   (ncc omitted when volumes absent).
    """
    import torch
    import torch.nn.functional as F

    R = _axis_angle_to_rotation_matrix(axis_angle)
    p_m = (fixed_mm_grid - c_f.view(1, 1, 1, 3) - delta.view(1, 1, 1, 3)) @ R + c_m.view(1, 1, 1, 3)
    grid = _mm_to_mov_grid_norm(p_m, mov_origin, mov_scales, mov_shape_hwz).unsqueeze(0)
    warped = F.grid_sample(
        mov_vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_mask = F.grid_sample(
        mov_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_n = F.normalize(warped, dim=1, eps=1e-6)
    fix_n = F.normalize(fix_vol, dim=1, eps=1e-6)
    cos = (warped_n * fix_n).sum(dim=1, keepdim=True)
    overlap = fix_mask * warped_mask
    denom = overlap.sum().clamp_min(1.0)
    cosine = (cos * overlap).sum() / denom

    score = cosine
    ncc = cosine.new_zeros(())
    if mov_int is not None and fix_int is not None and float(intensity_weight) > 0:
        warped_int = F.grid_sample(
            mov_int, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        ncc = _masked_intensity_ncc(warped_int, fix_int, overlap)
        score = cosine + float(intensity_weight) * ncc
    return score, cosine, ncc


def _adam_refine_semantic_pose(
    tensors,
    *,
    init_axis_angle,
    c_m,
    c_f,
    init_delta,
    n_steps,
    lr,
):
    """Adam on axis-angle + residual translation maximizing cosine (+ optional NCC)."""
    import torch

    device = tensors["mov_vol"].device
    axis_angle = torch.as_tensor(init_axis_angle, dtype=torch.float32, device=device).clone()
    axis_angle.requires_grad_(True)
    delta = torch.as_tensor(init_delta, dtype=torch.float32, device=device).clone()
    delta.requires_grad_(True)
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([axis_angle, delta], lr=float(lr))
    mov_int = tensors.get("mov_int")
    fix_int = tensors.get("fix_int")

    def _eval(aa, dlt):
        return _semantic_cosine_intensity_eval(
            tensors["mov_vol"],
            tensors["fix_vol"],
            tensors["fix_mask"],
            tensors["mov_mask"],
            tensors["fixed_mm"],
            tensors["mov_origin_t"],
            tensors["mov_scales_t"],
            tensors["mov_shape"],
            aa,
            c_m_t,
            c_f_t,
            dlt,
            mov_int=mov_int,
            fix_int=fix_int,
        )

    with torch.no_grad():
        init_score, init_cos, init_ncc = _eval(axis_angle.detach(), delta.detach())
        best = {
            "score": float(init_score.cpu()),
            "cosine": float(init_cos.cpu()),
            "ncc": float(init_ncc.cpu()),
            "init_score": float(init_score.cpu()),
            "init_cosine": float(init_cos.cpu()),
            "axis_angle": axis_angle.detach().clone(),
            "delta": delta.detach().clone(),
        }

    with torch.enable_grad():
        for _step in range(int(n_steps)):
            score, cosine, ncc = _eval(axis_angle, delta)
            loss = -score
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            score_val = float(score.detach().cpu())
            if score_val > best["score"]:
                best["score"] = score_val
                best["cosine"] = float(cosine.detach().cpu())
                best["ncc"] = float(ncc.detach().cpu())
                best["axis_angle"] = axis_angle.detach().clone()
                best["delta"] = delta.detach().clone()
    return best


def _rigid_from_octahedral_feature_probe(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    scan_name=None,
    debug=None,
):
    """
    Semantic octahedral orientation probe → 2-stage continuous refine.

    Evaluates the 24 proper rotations on downsampled joint-PCA feature volumes.
    Semantic correspondence (tissue-to-tissue cosine) is prioritized over
    binary mask Dice to prevent inverted/flipped fits when jaws or margins differ.
    Then refines pose via Adam on so(3) x R^3 coarse-to-fine (pool=2 -> pool=1).
    """
    import torch

    from .rigidAlignment import _rigid_rotation_angle_deg

    device = torch.device("cuda", 0)
    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)

    # Prepare features at OCTAHEDRAL_PROBE_POOL for rapid discrete rotation search
    probe_tensors = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        OCTAHEDRAL_PROBE_POOL,
        device,
    )
    if probe_tensors is None:
        if debug is not None:
            debug.meta["octahedral_probe"] = {"status": "probe_prep_failed"}
        return None

    probes = []
    best = None
    best_score = -999.0
    with torch.no_grad():
        for name, R in _octahedral_rotations():
            score_t, dice_t, cos_t = _semantic_dice_cosine_eval(
                probe_tensors["mov_vol"],
                probe_tensors["fix_vol"],
                probe_tensors["fix_mask"],
                probe_tensors["mov_mask"],
                probe_tensors["fixed_mm"],
                probe_tensors["mov_origin_t"],
                probe_tensors["mov_scales_t"],
                probe_tensors["mov_shape"],
                R,
                c_m_t,
                c_f_t,
            )
            score_val = float(score_t.cpu())
            dice_val = float(dice_t.cpu())
            cos_val = float(cos_t.cpu())
            angle = float(_rigid_rotation_angle_deg(R))
            probes.append({
                "name": name,
                "score": score_val,
                "cosine": cos_val,
                "dice": dice_val,
                "angle_deg": angle,
            })
            if score_val > best_score:
                best_score = score_val
                best = (name, R, score_val, dice_val, cos_val)

    del probe_tensors
    torch.cuda.empty_cache()

    identity_probe = next((p for p in probes if p["name"] == "x+y+z+"), None)
    id_score = identity_probe["score"] if identity_probe else -999.0

    record = {
        "probe_pool": int(OCTAHEDRAL_PROBE_POOL),
        "probes": probes,
        "identity_score": float(id_score),
        "best_name": None if best is None else best[0],
        "best_score": float(best_score),
        "metric": "semantic_cosine_gated_dice",
    }
    if debug is not None:
        debug.meta["octahedral_probe"] = record

    if best is None:
        return None

    name, R0, score0, dice0, cos0 = best
    angle0 = float(_rigid_rotation_angle_deg(R0))
    _log(
        scan_name,
        f"semantic octahedral probe: {name} score={score0:.3f} "
        f"(cos={cos0:.3f}, dice={dice0:.3f}) angle={angle0:.1f}°",
    )

    aa0 = _rotation_to_axis_angle(R0).astype(np.float32)
    delta0 = np.zeros(3, dtype=np.float32)

    # --- Stage 1: Coarse refinement at pool = 2 ---
    tensors_p2 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_COARSE_POOL,
        device,
    )
    if tensors_p2 is not None:
        hit1 = _adam_refine_hybrid_pose(
            tensors_p2,
            init_axis_angle=aa0,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta0,
            n_steps=SEMANTIC_REFINE_COARSE_STEPS,
            lr=SEMANTIC_REFINE_COARSE_LR,
        )
        aa_refined = hit1["axis_angle"].detach().cpu().numpy().astype(np.float32)
        delta_refined = hit1["delta"].detach().cpu().numpy().astype(np.float32)
        del tensors_p2
        torch.cuda.empty_cache()
    else:
        hit1 = None
        aa_refined = aa0
        delta_refined = delta0

    # --- Stage 2: Fine refinement at pool = 1 ---
    tensors_p1 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_FINE_POOL,
        device,
    )
    if tensors_p1 is not None:
        hit2 = _adam_refine_hybrid_pose(
            tensors_p1,
            init_axis_angle=aa_refined,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta_refined,
            n_steps=SEMANTIC_REFINE_FINE_STEPS,
            lr=SEMANTIC_REFINE_FINE_LR,
        )
        final_aa = hit2["axis_angle"]
        final_delta = hit2["delta"]
        final_score = hit2["score"]
        final_dice = hit2["dice"]
        final_cos = hit2["cosine"]
        del tensors_p1
        torch.cuda.empty_cache()
    elif hit1 is not None:
        final_aa = hit1["axis_angle"]
        final_delta = hit1["delta"]
        final_score = hit1["score"]
        final_dice = hit1["dice"]
        final_cos = hit1["cosine"]
    else:
        final_aa = torch.as_tensor(aa0, dtype=torch.float32, device=device)
        final_delta = torch.as_tensor(delta0, dtype=torch.float32, device=device)
        final_score = score0
        final_dice = dice0
        final_cos = cos0

    R = _axis_angle_to_rotation_matrix(final_aa).detach().cpu().numpy().astype(np.float64)
    delta = final_delta.detach().cpu().numpy().astype(np.float64).reshape(3)
    t = c_f + delta - R @ c_m
    angle = float(_rigid_rotation_angle_deg(R))

    record["status"] = "accepted"
    record["probe_name"] = name
    record["probe_score"] = float(score0)
    record["probe_dice"] = float(dice0)
    record["probe_cosine"] = float(cos0)
    record["probe_angle_deg"] = angle0
    record["refined_dice"] = float(final_dice)
    record["refined_cosine"] = float(final_cos)
    record["refined_score"] = float(final_score)
    record["refined_angle_deg"] = angle
    record["coarse_steps"] = int(SEMANTIC_REFINE_COARSE_STEPS)
    record["fine_steps"] = int(SEMANTIC_REFINE_FINE_STEPS)

    _log(
        scan_name,
        f"semantic 2-stage refine: dice {dice0:.3f}→{final_dice:.3f} "
        f"cos {cos0:.3f}→{final_cos:.3f} score {score0:.3f}→{final_score:.3f} "
        f"angle={angle:.1f}° (coarse={SEMANTIC_REFINE_COARSE_STEPS}st fine={SEMANTIC_REFINE_FINE_STEPS}st)",
    )
    if debug is not None:
        debug.meta["octahedral_probe"] = record
        debug.meta["accepted_pose"] = {
            "R": R.tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "angle_deg": angle,
            "label": f"semantic_octahedral:{name}",
            "source": "semantic_octahedral_refine",
            "probe_dice": float(dice0),
            "probe_cosine": float(cos0),
            "refined_dice": float(final_dice),
            "refined_cosine": float(final_cos),
            "refined_score": float(final_score),
        }
    return _dino_reg_pose(
        R, t, c_m, c_f, "semantic_octahedral_refine",
        angle_deg=angle, probe_name=name,
        refined_dice=float(final_dice),
        refined_cosine=float(final_cos),
    )


def _feature_map_cosine_at_pose(
    mov_vol,
    fix_vol,
    fix_mask,
    mov_mask,
    fixed_mm_grid,
    mov_origin,
    mov_scales,
    mov_shape_hwz,
    axis_angle,
    c_m,
    c_f,
    delta,
):
    """
    Masked mean cosine after warping moving features onto the fixed grid.

    Rotation is about the moving centroid, then residual translation ``delta``:

        p_fixed = R @ (p_moving - c_m) + c_f + delta

    Inverse sampling (what ``grid_sample`` needs):

        p_moving = R.T @ (p_fixed - c_f - delta) + c_m
    """
    import torch
    import torch.nn.functional as F

    R = _axis_angle_to_rotation_matrix(axis_angle)
    p_m = (fixed_mm_grid - c_f.view(1, 1, 1, 3) - delta.view(1, 1, 1, 3)) @ R + c_m.view(1, 1, 1, 3)
    grid = _mm_to_mov_grid_norm(p_m, mov_origin, mov_scales, mov_shape_hwz).unsqueeze(0)
    warped = F.grid_sample(
        mov_vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_mask = F.grid_sample(
        mov_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_n = F.normalize(warped, dim=1, eps=1e-6)
    fix_n = F.normalize(fix_vol, dim=1, eps=1e-6)
    cos = (warped_n * fix_n).sum(dim=1, keepdim=True)
    overlap = fix_mask * warped_mask
    denom = overlap.sum().clamp_min(1.0)
    return (cos * overlap).sum() / denom, warped


def _prepare_feature_map_volumes(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    pool,
    device,
    mov_intensity=None,
    fix_intensity=None,
):
    """Pool PCA grids and build torch volumes + fixed-mm coordinate grid."""
    import torch

    mov_f, mov_m, mov_mult = _pool_feature_level(mov_feat, moving_mask_feat, pool)
    fix_f, fix_m, fix_mult = _pool_feature_level(fix_feat, fixed_mask_feat, pool)
    mov_scales_p = np.asarray(mov_scales, dtype=np.float64) * float(mov_mult)
    fix_scales_p = np.asarray(fix_scales, dtype=np.float64) * float(fix_mult)
    if int(mov_m.sum()) < 32 or int(fix_m.sum()) < 32:
        return None

    mov_vol = _feat_hwzd_to_torch(mov_f).to(device)
    fix_vol = _feat_hwzd_to_torch(fix_f).to(device)
    fix_mask = _mask_hwz_to_torch(fix_m, device)
    mov_mask = _mask_hwz_to_torch(mov_m, device)
    mov_shape = mov_f.shape[:3]
    fixed_mm = _fixed_mm_grid(fix_origin, fix_scales_p, fix_f.shape[:3], device)
    mov_origin_t = torch.as_tensor(mov_origin, dtype=torch.float32, device=device)
    mov_scales_t = torch.as_tensor(mov_scales_p, dtype=torch.float32, device=device)
    out = {
        "mov_vol": mov_vol,
        "fix_vol": fix_vol,
        "fix_mask": fix_mask,
        "mov_mask": mov_mask,
        "fixed_mm": fixed_mm,
        "mov_origin_t": mov_origin_t,
        "mov_scales_t": mov_scales_t,
        "mov_shape": mov_shape,
        "pool": int(pool),
    }
    if mov_intensity is not None and fix_intensity is not None:
        mov_i = _resize_intensity_to_feat(mov_intensity, mov_feat.shape[:3])
        fix_i = _resize_intensity_to_feat(fix_intensity, fix_feat.shape[:3])
        mov_i_p, _, _ = _pool_feature_level(
            mov_i[..., None], moving_mask_feat, pool
        )
        fix_i_p, _, _ = _pool_feature_level(
            fix_i[..., None], fixed_mask_feat, pool
        )
        out["mov_int"] = _feat_hwzd_to_torch(mov_i_p).to(device)
        out["fix_int"] = _feat_hwzd_to_torch(fix_i_p).to(device)
    return out


def _resize_intensity_to_feat(intensity, feat_shape_hwz):
    """Resize a normalized intensity crop to the feature grid (H, W, D)."""
    vol = np.asarray(intensity, dtype=np.float32)
    th, tw, td = (int(feat_shape_hwz[0]), int(feat_shape_hwz[1]), int(feat_shape_hwz[2]))
    if vol.shape == (th, tw, td):
        return vol
    return resize(
        vol, (th, tw, td), anti_aliasing=False, preserve_range=True
    ).astype(np.float32)


def _adam_refine_feature_map_pose(
    tensors,
    *,
    init_axis_angle,
    c_m,
    c_f,
    init_delta,
    n_steps,
    lr,
):
    """Adam on axis-angle + residual centroid translation; returns best pose."""
    import torch

    device = tensors["mov_vol"].device
    axis_angle = torch.as_tensor(init_axis_angle, dtype=torch.float32, device=device).clone()
    axis_angle.requires_grad_(True)
    delta = torch.as_tensor(init_delta, dtype=torch.float32, device=device).clone()
    delta.requires_grad_(True)
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([axis_angle, delta], lr=float(lr))

    def _eval(aa, dlt):
        return _feature_map_cosine_at_pose(
            tensors["mov_vol"],
            tensors["fix_vol"],
            tensors["fix_mask"],
            tensors["mov_mask"],
            tensors["fixed_mm"],
            tensors["mov_origin_t"],
            tensors["mov_scales_t"],
            tensors["mov_shape"],
            aa,
            c_m_t,
            c_f_t,
            dlt,
        )

    with torch.no_grad():
        init_cos, _ = _eval(axis_angle.detach(), delta.detach())
        best = {
            "cosine": float(init_cos.cpu()),
            "init_cosine": float(init_cos.cpu()),
            "axis_angle": axis_angle.detach().clone(),
            "delta": delta.detach().clone(),
        }

    with torch.enable_grad():
        for _step in range(int(n_steps)):
            cos_mean, _warped = _eval(axis_angle, delta)
            loss = -cos_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cos_val = float(cos_mean.detach().cpu())
            if cos_val > best["cosine"]:
                best["cosine"] = cos_val
                best["axis_angle"] = axis_angle.detach().clone()
                best["delta"] = delta.detach().clone()
    return best


def _optimize_rigid_on_feature_maps(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    *,
    scan_name=None,
):
    """
    Multi-start + coarse-to-fine GD on joint-PCA feature maps.

    Rotation is about the moving centroid so ±90° seeds stay overlapping.
    Returns (R_col, t_mm, record) with ``p_fixed = R_col @ p_moving + t``.
    """
    import torch

    from .rigidAlignment import _rigid_rotation_angle_deg

    device = torch.device("cuda", 0)
    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    zero_delta = np.zeros(3, dtype=np.float32)
    orientation_seeds = _orientation_seed_axis_angles()
    multistart_log = []
    best_aa = None
    best_delta = None
    best_cos = -1.0
    best_pool = 1

    last_pool = 1
    last_cos = -1.0
    pool_best_cos = -1.0

    for pool in FEATURE_MAP_OPT_POOLS:
        n_steps = (
            FEATURE_MAP_OPT_STEPS_COARSE if int(pool) >= 2 else FEATURE_MAP_OPT_STEPS_FINE
        )
        tensors = _prepare_feature_map_volumes(
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            pool,
            device,
        )
        if tensors is None:
            continue

        if int(pool) >= 4:
            seeds = orientation_seeds
        else:
            aa_seed = (
                best_aa.detach().cpu().numpy()
                if best_aa is not None
                else np.zeros(3, dtype=np.float32)
            )
            seeds = (("warm", tuple(float(x) for x in aa_seed)),)

        pool_best_cos = -1.0
        pool_best_aa = pool_best_delta = None
        for seed_name, aa_seed in seeds:
            dlt_seed = (
                best_delta.detach().cpu().numpy()
                if best_delta is not None and seed_name == "warm"
                else zero_delta
            )
            hit = _adam_refine_feature_map_pose(
                tensors,
                init_axis_angle=np.asarray(aa_seed, dtype=np.float32),
                c_m=c_m,
                c_f=c_f,
                init_delta=np.asarray(dlt_seed, dtype=np.float32),
                n_steps=n_steps,
                lr=FEATURE_MAP_OPT_LR,
            )
            R_col = _axis_angle_to_rotation_matrix(hit["axis_angle"]).detach().cpu().numpy()
            angle = float(_rigid_rotation_angle_deg(R_col))
            cos_val = float(hit["cosine"])
            multistart_log.append({
                "pool": int(pool),
                "seed": seed_name,
                "init_cosine": float(hit["init_cosine"]),
                "cosine": cos_val,
                "angle_deg": angle,
            })
            if cos_val > pool_best_cos:
                pool_best_cos = cos_val
                pool_best_aa = hit["axis_angle"]
                pool_best_delta = hit["delta"]
            if cos_val > best_cos:
                best_cos = cos_val
                best_aa = hit["axis_angle"]
                best_delta = hit["delta"]
                best_pool = int(pool)

        if pool_best_aa is not None:
            best_aa = pool_best_aa
            best_delta = pool_best_delta
            last_pool = int(pool)
            last_cos = float(pool_best_cos)
            _log(
                scan_name,
                f"feature-map pool ÷{pool}: best cosine={pool_best_cos:.3f} "
                f"({len(seeds)} seed(s))",
            )

    if best_aa is None or best_delta is None:
        return None, None, {"status": "opt_failed", "multistart": multistart_log}

    R_col = _axis_angle_to_rotation_matrix(best_aa).detach().cpu().numpy().astype(np.float64)
    delta = best_delta.detach().cpu().numpy().astype(np.float64)
    t_mm = c_f + delta - R_col @ c_m
    report_pool = int(last_pool)
    report_cos = float(last_cos) if last_cos >= 0 else float(best_cos)

    id_cos = -1.0
    fine_tensors = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        report_pool,
        device,
    )
    if fine_tensors is not None:
        id_aa = torch.zeros(3, device=device, dtype=torch.float32)
        id_dlt = torch.zeros(3, device=device, dtype=torch.float32)
        c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
        c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)
        id_cos_t, _ = _feature_map_cosine_at_pose(
            fine_tensors["mov_vol"],
            fine_tensors["fix_vol"],
            fine_tensors["fix_mask"],
            fine_tensors["mov_mask"],
            fine_tensors["fixed_mm"],
            fine_tensors["mov_origin_t"],
            fine_tensors["mov_scales_t"],
            fine_tensors["mov_shape"],
            id_aa,
            c_m_t,
            c_f_t,
            id_dlt,
        )
        id_cos = float(id_cos_t.detach().cpu())

    record = {
        "status": "ok",
        "pool": int(report_pool),
        "steps_coarse": int(FEATURE_MAP_OPT_STEPS_COARSE),
        "steps_fine": int(FEATURE_MAP_OPT_STEPS_FINE),
        "cosine_opt": float(report_cos),
        "cosine_identity": float(id_cos),
        "cosine_gain": float(report_cos - id_cos),
        "multistart": multistart_log,
        "n_seeds": int(len(orientation_seeds)),
    }
    return R_col, t_mm, record


def _rigid_from_feature_map_optimization(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    scan_name=None,
    debug=None,
):
    """
    Primary DINO-Reg rigid: gradient descent on joint-PCA regional feature maps.

    Semantic feature volume → semantic feature volume registration; returns a
    physical rigid pose applied directly to the intensity volume (no guidepoints).
    """
    from .rigidAlignment import _rigid_rotation_angle_deg

    record = {"status": "started"}
    R_col, t_mm, opt_record = _optimize_rigid_on_feature_maps(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        c_m,
        c_f,
        scan_name=scan_name,
    )
    record.update(opt_record or {})
    if R_col is None:
        _log(scan_name, f"feature-map optimizer: {record.get('status', 'failed')}")
        if debug is not None:
            debug.meta["feature_map_opt"] = record
        return None

    R_col = np.asarray(R_col, dtype=np.float64)
    angle = float(_rigid_rotation_angle_deg(R_col))
    gain = float(record.get("cosine_gain", 0.0))
    id_cos = float(record.get("cosine_identity", 0.0))
    opt_cos = float(record.get("cosine_opt", 0.0))
    record["angle_deg"] = angle
    record["R"] = R_col.tolist()
    record["t"] = np.asarray(t_mm, dtype=np.float64).tolist()

    if debug is not None:
        debug.meta["feature_map_opt"] = record
        try:
            debug.save_feature_map_opt_summary(record)
        except Exception:
            pass

    # Identity (or near-identity) with almost no cosine gain is just centroid paste.
    if angle < 8.0 and gain < float(FEATURE_MAP_OPT_MIN_COSINE_GAIN):
        _log(
            scan_name,
            f"feature-map optimizer rejected: near-identity gain={gain:.4f} "
            f"(id={id_cos:.3f} opt={opt_cos:.3f}) angle={angle:.1f}°",
        )
        record["status"] = "identity_guard"
        if debug is not None:
            debug.meta["feature_map_opt"] = record
        return None

    if opt_cos < id_cos + 1e-4 and angle >= 8.0:
        _log(
            scan_name,
            f"feature-map optimizer rejected: rotation does not beat identity "
            f"(id={id_cos:.3f} opt={opt_cos:.3f}) angle={angle:.1f}°",
        )
        record["status"] = "identity_guard"
        if debug is not None:
            debug.meta["feature_map_opt"] = record
        return None

    if opt_cos < float(FEATURE_MAP_OPT_MIN_ABSOLUTE_COSINE):
        _log(
            scan_name,
            f"feature-map optimizer rejected: absolute cosine {opt_cos:.3f} "
            f"< {FEATURE_MAP_OPT_MIN_ABSOLUTE_COSINE} (features still misaligned)",
        )
        record["status"] = "low_absolute_cosine"
        if debug is not None:
            debug.meta["feature_map_opt"] = record
        return None

    _log(
        scan_name,
        f"feature-map optimizer accepted: cosine {id_cos:.3f}→{opt_cos:.3f} "
        f"(+{gain:.3f}) angle={angle:.1f}° pool=÷{record.get('pool', 1)}",
    )
    record["status"] = "accepted"
    if debug is not None:
        debug.meta["accepted_pose"] = {
            "R": R_col.tolist(),
            "t": np.asarray(t_mm, dtype=np.float64).tolist(),
            "angle_deg": angle,
            "label": "feature_map_opt",
            "source": "feature_map_opt",
            "cosine_gain": gain,
        }
    return _dino_reg_pose(
        R_col,
        t_mm,
        c_m,
        c_f,
        "feature_map_opt",
        angle_deg=angle,
        cosine_gain=gain,
    )


def _debug_native_overlay_geom(
    mov_mask_native,
    fix_mask_native,
    mov_vs,
    fix_vs,
    z0_m,
    z0_f,
):
    """Native-crop masks + mm scales/origins for debug overlays, or all-None."""
    if mov_mask_native is None or fix_mask_native is None or not mov_vs or not fix_vs:
        return None, None, None, None, None, None
    dbg_mov_mask = np.asarray(mov_mask_native, dtype=bool)
    dbg_fix_mask = np.asarray(fix_mask_native, dtype=bool)
    dbg_mov_scales = np.array([float(mov_vs)] * 3, dtype=np.float64)
    dbg_fix_scales = np.array([float(fix_vs)] * 3, dtype=np.float64)
    dbg_mov_origin = np.array([0.0, 0.0, float(z0_m) * float(mov_vs)], dtype=np.float64)
    dbg_fix_origin = np.array([0.0, 0.0, float(z0_f) * float(fix_vs)], dtype=np.float64)
    return (
        dbg_mov_mask,
        dbg_fix_mask,
        dbg_mov_origin,
        dbg_mov_scales,
        dbg_fix_origin,
        dbg_fix_scales,
    )


def _adam_refine_from_coarse_rt(
    R0,
    t0,
    c_m,
    c_f,
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    mov_intensity=None,
    fix_intensity=None,
    scan_name=None,
    debug=None,
    overlay_geom=None,
    source_label="mutual_nn",
    overlay_prefix="",
):
    """
    Coarse→fine Adam on axis-angle + residual translation (feature cosine + NCC).

    ``overlay_geom`` is an optional 6-tuple of native masks/origins/scales for
    debug mesh overlays; when None, refine overlays are skipped.
    ``overlay_prefix`` distinguishes octahedral vs cluster refine PNGs.
    Returns ``(R, t, final_cos, final_score, angle0, angle)``.
    """
    import torch

    from .rigidAlignment import _rigid_rotation_angle_deg

    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(t0, dtype=np.float64).reshape(3)
    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    angle0 = float(_rigid_rotation_angle_deg(R0))
    device = torch.device("cuda", 0)
    aa0 = _rotation_to_axis_angle(R0).astype(np.float32)
    delta0 = (t0 - c_f + R0 @ c_m).astype(np.float32)

    hit1 = None
    tensors_p2 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_COARSE_POOL,
        device,
        mov_intensity=mov_intensity,
        fix_intensity=fix_intensity,
    )
    if tensors_p2 is not None:
        hit1 = _adam_refine_semantic_pose(
            tensors_p2,
            init_axis_angle=aa0,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta0,
            n_steps=SEMANTIC_REFINE_COARSE_STEPS,
            lr=SEMANTIC_REFINE_COARSE_LR,
        )
        aa_r = hit1["axis_angle"].detach().cpu().numpy().astype(np.float32)
        delta_r = hit1["delta"].detach().cpu().numpy().astype(np.float32)
        if debug is not None and overlay_geom is not None:
            (
                ov_mov_mask,
                ov_fix_mask,
                ov_mov_origin,
                ov_mov_scales,
                ov_fix_origin,
                ov_fix_scales,
            ) = overlay_geom
            R_c = _axis_angle_to_rotation_matrix(hit1["axis_angle"]).detach().cpu().numpy()
            t_c = c_f + delta_r.astype(np.float64) - R_c @ c_m
            debug.save_rigid_mesh_overlay(
                ov_mov_mask,
                ov_fix_mask,
                ov_mov_origin,
                ov_mov_scales,
                ov_fix_origin,
                ov_fix_scales,
                R_c,
                t_c,
                filename=f"overlay_{overlay_prefix}refine_coarse.png",
                title=(
                    f"refine coarse (pool÷{SEMANTIC_REFINE_COARSE_POOL})  "
                    f"cos={float(hit1['cosine']):.3f}"
                ),
            )
        del tensors_p2
        torch.cuda.empty_cache()
    else:
        aa_r, delta_r = aa0, delta0

    tensors_p1 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_FINE_POOL,
        device,
        mov_intensity=mov_intensity,
        fix_intensity=fix_intensity,
    )
    if tensors_p1 is not None:
        hit2 = _adam_refine_semantic_pose(
            tensors_p1,
            init_axis_angle=aa_r,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta_r,
            n_steps=SEMANTIC_REFINE_FINE_STEPS,
            lr=SEMANTIC_REFINE_FINE_LR,
        )
        final_aa = hit2["axis_angle"]
        final_delta = hit2["delta"]
        final_cos = float(hit2["cosine"])
        final_score = float(hit2["score"])
        del tensors_p1
        torch.cuda.empty_cache()
    elif hit1 is not None:
        final_aa = hit1["axis_angle"]
        final_delta = hit1["delta"]
        final_cos = float(hit1["cosine"])
        final_score = float(hit1["score"])
    else:
        final_aa = torch.as_tensor(aa0, dtype=torch.float32, device=device)
        final_delta = torch.as_tensor(delta0, dtype=torch.float32, device=device)
        final_cos = float("nan")
        final_score = float("nan")

    R = _axis_angle_to_rotation_matrix(final_aa).detach().cpu().numpy().astype(np.float64)
    delta = final_delta.detach().cpu().numpy().astype(np.float64).reshape(3)
    t = c_f + delta - R @ c_m
    angle = float(_rigid_rotation_angle_deg(R))
    _log(
        scan_name,
        f"{source_label} refine: cos={final_cos:.3f} score={final_score:.3f} "
        f"angle={angle0:.1f}°→{angle:.1f}°",
    )
    if debug is not None and overlay_geom is not None:
        (
            ov_mov_mask,
            ov_fix_mask,
            ov_mov_origin,
            ov_mov_scales,
            ov_fix_origin,
            ov_fix_scales,
        ) = overlay_geom
        debug.save_rigid_mesh_overlay(
            ov_mov_mask,
            ov_fix_mask,
            ov_mov_origin,
            ov_mov_scales,
            ov_fix_origin,
            ov_fix_scales,
            R,
            t,
            filename=f"overlay_{overlay_prefix}refine_fine.png",
            title=(
                f"refine fine (pool÷{SEMANTIC_REFINE_FINE_POOL})  "
                f"cos={final_cos:.3f}  angle={angle:.1f}°"
            ),
        )
    return R, t, final_cos, final_score, angle0, angle


def _rigid_from_mutual_nn_and_refine(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    scan_name=None,
    debug=None,
    mov_intensity=None,
    fix_intensity=None,
    mov_mask_native=None,
    fix_mask_native=None,
    mov_vs=None,
    fix_vs=None,
    z0_m=0,
    z0_f=0,
):
    """
    Primary pose: pool÷2 mutual-NN feature matches → RANSAC → Adam refine.

    Each foreground token is a 24-D PCA feature at a physical mm location. Different
    volume sizes are fine — we match point clouds, not equal grids. Mutual NN +
    Lowe ratio keeps only confident 1-to-1 pairs; RANSAC rejects outliers; Adam
    then maximizes dense feature cosine.
    """
    from .rigidAlignment import _rigid_rotation_angle_deg

    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    overlay_geom = _debug_native_overlay_geom(
        mov_mask_native, fix_mask_native, mov_vs, fix_vs, z0_m, z0_f
    )
    if overlay_geom[0] is None:
        overlay_geom = (
            np.asarray(moving_mask_feat, dtype=bool),
            np.asarray(fixed_mask_feat, dtype=bool),
            mov_origin,
            mov_scales,
            fix_origin,
            fix_scales,
        )

    best = None
    best_pick = (-1.0, -1, -999)  # (weighted, n_in, -pool)
    best_corr = None
    for pool in MUTUAL_NN_COARSE_POOLS:
        _log(
            scan_name,
            f"mutual-NN coarse: trying pool÷{pool} "
            f"(cosine threshold + mutual NN → RANSAC)",
        )
        hit = _try_rigid_at_pyramid_level(
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            c_m,
            c_f,
            mov_diag_mm,
            fix_diag_mm,
            pool,
            scan_name=scan_name,
            debug=debug,
        )
        if hit is None:
            continue
        R_lvl, t_lvl, label, score_tuple, corr = hit
        weighted, n_in, _pool, angle = score_tuple
        pick = (float(weighted), int(n_in), -int(_pool))
        if pick > best_pick:
            best_pick = pick
            best = (
                R_lvl, t_lvl, label, int(_pool), int(n_in), float(angle), float(weighted)
            )
            best_corr = corr

    if best is None:
        _log(
            scan_name,
            "mutual-NN coarse: no acceptable rigid at pool "
            f"{list(MUTUAL_NN_COARSE_POOLS)}",
        )
        if debug is not None:
            debug.meta["mutual_nn_refine"] = {
                "status": "coarse_failed",
                "pools": list(MUTUAL_NN_COARSE_POOLS),
            }
        return None

    R0, t0, label, pool, n_in, angle_c, weighted = best
    angle0 = float(_rigid_rotation_angle_deg(R0))
    _log(
        scan_name,
        f"mutual-NN coarse: accepted pool÷{pool} ({label}) "
        f"angle={angle0:.1f}° inliers={n_in} score={weighted:.1f}",
    )
    if debug is not None:
        debug.meta["mutual_nn_refine"] = {
            "status": "coarse_ok",
            "pool": int(pool),
            "label": str(label),
            "n_inliers": int(n_in),
            "score": float(weighted),
            "coarse_angle_deg": angle0,
            "pools_tried": list(MUTUAL_NN_COARSE_POOLS),
            "n_match_landmarks": int(len(best_corr["mov_pts"])) if best_corr else 0,
            "match_used_inliers": bool(best_corr.get("used_inliers")) if best_corr else False,
        }
        debug.save_rigid_mesh_overlay(
            overlay_geom[0],
            overlay_geom[1],
            overlay_geom[2],
            overlay_geom[3],
            overlay_geom[4],
            overlay_geom[5],
            R0,
            t0,
            filename="overlay_best_kabsch.png",
            title=f"mutual-NN pool÷{pool}  angle={angle0:.1f}°  in={n_in}",
            mov_centroids_mm=None if best_corr is None else best_corr["mov_pts"],
            fix_centroids_mm=None if best_corr is None else best_corr["fix_pts"],
        )
        if best_corr is not None:
            debug.save_match_landmarks_side_by_side(
                overlay_geom[0],
                overlay_geom[1],
                overlay_geom[2],
                overlay_geom[3],
                overlay_geom[4],
                overlay_geom[5],
                R0,
                t0,
                best_corr["mov_pts"],
                best_corr["fix_pts"],
                scores=best_corr["scores"],
                filename="overlay_best_matches_side_by_side.png",
                title=(
                    f"mutual-NN pool÷{pool} landmarks  "
                    f"angle={angle0:.1f}°  in={n_in}"
                ),
            )
            debug.save_landmark_figure(
                best_corr["mov_pts"],
                best_corr["fix_pts"],
                best_corr["scores"],
                status=f"mutual-NN pool÷{pool} inliers",
            )

    R, t, final_cos, final_score, angle0, angle = _adam_refine_from_coarse_rt(
        R0,
        t0,
        c_m,
        c_f,
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        mov_intensity=mov_intensity,
        fix_intensity=fix_intensity,
        scan_name=scan_name,
        debug=debug,
        overlay_geom=overlay_geom,
        source_label=f"mutual-NN÷{pool}",
    )
    if debug is not None:
        debug.meta["mutual_nn_refine"].update({
            "status": "accepted",
            "refined_cosine": final_cos,
            "refined_score": final_score,
            "refined_angle_deg": angle,
        })
        debug.meta["accepted_pose"] = {
            "R": R.tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "angle_deg": angle,
            "label": f"mutual_nn_pool{pool}",
            "probe_name": f"mutual_nn÷{pool}",
            "source": "mutual_nn_refine",
            "refined_cosine": final_cos,
        }
    return _dino_reg_pose(
        R,
        t,
        c_m,
        c_f,
        "mutual_nn_refine",
        label=f"mutual_nn_pool{pool}",
        probe_name=f"mutual_nn÷{pool}",
        angle_deg=angle,
        refined_cosine=final_cos,
        n_inliers=int(n_in),
    )


def _rigid_from_semantic_landmarks_and_refine(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    scan_name=None,
    debug=None,
    mov_intensity=None,
    fix_intensity=None,
    mov_mask_native=None,
    fix_mask_native=None,
    mov_vs=None,
    fix_vs=None,
    z0_m=0,
    z0_f=0,
    max_residual_angle_deg=None,
):
    """
    Multi-K semantic-cluster CoM landmarks → weighted Kabsch → cosine Adam refine.

    ---------------------------------------------------------------------------
    REVERT / PROVEN FALLBACK
    ---------------------------------------------------------------------------
    This was the primary coarse pose for a long stretch and worked *very well*
    for most Tongue_fat / KDO subjects: multi-K joint k-means → cluster CoMs →
    Kabsch, ranked by masked feature cosine, then Adam refine. Even with a
    known midline / "onion-layer" bias in the CoMs (concentric tissue shells
    pull centroids toward the long axis, leaving rotation about that axis
    under-constrained), the majority of scans still landed inside the Adam
    capture basin and refined cleanly.

    Failures (e.g. KDO75, KDO144) showed all K trials stuck near ~90–110° with
    collinear midline landmarks. The main path now starts with a cheap
    octahedral encode search; this function is the fallback when that cube is
    a weak winner or Adam cosine stays in the failure band.
    ---------------------------------------------------------------------------
    """
    import torch

    from .rigidAlignment import _rigid_rotation_angle_deg

    c_m = np.asarray(c_m, dtype=np.float64).reshape(3)
    c_f = np.asarray(c_f, dtype=np.float64).reshape(3)
    nested_fg = _is_nested_foreground(mov_diag_mm, fix_diag_mm)
    inlier_mm = max(4.0, 0.16 * min(float(mov_diag_mm), float(fix_diag_mm)))
    if nested_fg:
        inlier_mm = max(4.0, 0.22 * float(mov_diag_mm))
    spread_mm = max(1.5, 0.08 * min(float(mov_diag_mm), float(fix_diag_mm)))

    # Prefer native-resolution masks for debug meshes when available.
    if mov_mask_native is not None and fix_mask_native is not None and mov_vs and fix_vs:
        dbg_mov_mask = np.asarray(mov_mask_native, dtype=bool)
        dbg_fix_mask = np.asarray(fix_mask_native, dtype=bool)
        dbg_mov_scales = np.array([float(mov_vs)] * 3, dtype=np.float64)
        dbg_fix_scales = np.array([float(fix_vs)] * 3, dtype=np.float64)
        dbg_mov_origin = np.array([0.0, 0.0, float(z0_m) * float(mov_vs)], dtype=np.float64)
        dbg_fix_origin = np.array([0.0, 0.0, float(z0_f) * float(fix_vs)], dtype=np.float64)
    else:
        dbg_mov_mask = dbg_fix_mask = None
        dbg_mov_scales = dbg_fix_scales = None
        dbg_mov_origin = dbg_fix_origin = None

    device = torch.device("cuda", 0)
    rank_tensors = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_K_RANK_POOL,
        device,
    )
    c_m_t = torch.as_tensor(c_m, dtype=torch.float32, device=device)
    c_f_t = torch.as_tensor(c_f, dtype=torch.float32, device=device)

    trials = []
    trial_overlays = []
    best_pick = None
    best_key = (-999.0, -999.0)  # (cosine, -med_res)
    best_packed = None
    best_record = {"status": "not_started"}

    id_R = np.eye(3, dtype=np.float64)
    id_t = c_f - c_m
    delta_id = (id_t - c_f + id_R @ c_m).astype(np.float32)
    cos_id = float("nan")
    if rank_tensors is not None:
        with torch.no_grad():
            cos_id_t = _feature_map_cosine_given_R(
                rank_tensors["mov_vol"],
                rank_tensors["fix_vol"],
                rank_tensors["fix_mask"],
                rank_tensors["mov_mask"],
                rank_tensors["fixed_mm"],
                rank_tensors["mov_origin_t"],
                rank_tensors["mov_scales_t"],
                rank_tensors["mov_shape"],
                id_R,
                c_m_t,
                c_f_t,
                torch.as_tensor(delta_id, dtype=torch.float32, device=device),
            )
            cos_id = float(cos_id_t.detach().cpu())
    if np.isfinite(cos_id):
        best_key = (cos_id, 0.0)
        best_pick = {
            "R": id_R,
            "t": id_t,
            "label": "identity",
            "angle_deg": 0.0,
            "n_inliers": 0,
            "median_residual_mm": 1e9,
            "estimator": "identity_baseline",
        }
        best_record = {
            "k": 0,
            "status": "identity_baseline",
            "cosine": cos_id,
            **best_pick,
        }

    _log(
        scan_name,
        f"semantic landmarks: trying Kabsch over K={list(SEMANTIC_LANDMARK_K_TRIES)} "
        f"(rank by feature cosine @ pool÷{SEMANTIC_K_RANK_POOL}, baseline identity cos={cos_id:.3f})",
    )

    for k_try in SEMANTIC_LANDMARK_K_TRIES:
        packed, record = _semantic_landmarks_from_clusters(
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            k=k_try,
        )
        n_lm = int(record.get("n_landmarks", 0) or 0)
        trial = {
            "k": int(k_try),
            "landmark_status": record.get("status"),
            "n_landmarks": n_lm,
        }
        if packed is None:
            _log(
                scan_name,
                f"semantic landmarks K={k_try}: skipped ({record.get('status')}, "
                f"landmarks={n_lm})",
            )
            trials.append(trial)
            continue
        moving_corr, fixed_corr, scores, mov_grid, fix_grid, mask_m, mask_f = packed
        ov_mov_mask = dbg_mov_mask if dbg_mov_mask is not None else mask_m
        ov_fix_mask = dbg_fix_mask if dbg_fix_mask is not None else mask_f
        ov_mov_origin = dbg_mov_origin if dbg_mov_origin is not None else mov_origin
        ov_fix_origin = dbg_fix_origin if dbg_fix_origin is not None else fix_origin
        ov_mov_scales = dbg_mov_scales if dbg_mov_scales is not None else mov_scales
        ov_fix_scales = dbg_fix_scales if dbg_fix_scales is not None else fix_scales
        min_inliers = max(MIN_LANDMARKS, len(moving_corr) // 2)
        pose = _pick_rigid_from_landmark_pairs(
            moving_corr,
            fixed_corr,
            scores,
            c_m,
            c_f,
            inlier_mm,
            spread_mm,
            min_inliers,
            scan_name=scan_name,
            stage_label=f"semantic-landmarks K={k_try}",
            max_angle_deg=max_residual_angle_deg,
        )
        if pose is None:
            trial["pose_status"] = "rejected"
            _log(
                scan_name,
                f"semantic landmarks K={k_try}: Kabsch rejected "
                f"(landmarks={len(moving_corr)})",
            )
            trials.append(trial)
            continue
        n_in = int(pose["n_inliers"])
        med = float(pose["median_residual_mm"])
        R_pose = np.asarray(pose["R"], dtype=np.float64)
        t_pose = np.asarray(pose["t"], dtype=np.float64).reshape(3)
        # p_fixed = R @ p_moving + t  ⇔  delta = t - c_f + R @ c_m in centroid form
        delta_pose = (t_pose - c_f + R_pose @ c_m).astype(np.float32)
        cosine = float("nan")
        if rank_tensors is not None:
            with torch.no_grad():
                cos_t = _feature_map_cosine_given_R(
                    rank_tensors["mov_vol"],
                    rank_tensors["fix_vol"],
                    rank_tensors["fix_mask"],
                    rank_tensors["mov_mask"],
                    rank_tensors["fixed_mm"],
                    rank_tensors["mov_origin_t"],
                    rank_tensors["mov_scales_t"],
                    rank_tensors["mov_shape"],
                    R_pose,
                    c_m_t,
                    c_f_t,
                    torch.as_tensor(delta_pose, dtype=torch.float32, device=device),
                )
                cosine = float(cos_t.detach().cpu())
        # Rank by tissue cosine (K-independent); residual is only a tie-break.
        key = (cosine if np.isfinite(cosine) else -999.0, -med)
        trial.update({
            "pose_status": "ok",
            "n_inliers": n_in,
            "median_residual_mm": med,
            "angle_deg": float(pose["angle_deg"]),
            "label": str(pose["label"]),
            "estimator": str(pose.get("estimator", "weighted_kabsch")),
            "cosine": cosine,
        })
        trials.append(trial)
        _log(
            scan_name,
            f"semantic landmarks K={k_try}: Kabsch ok "
            f"landmarks={len(moving_corr)} inliers={n_in} "
            f"med_res={med:.2f} mm angle={float(pose['angle_deg']):.1f}° "
            f"cosine={cosine:.3f} label={pose['label']}",
        )
        if debug is not None:
            debug.save_rigid_mesh_overlay(
                ov_mov_mask,
                ov_fix_mask,
                ov_mov_origin,
                ov_mov_scales,
                ov_fix_origin,
                ov_fix_scales,
                pose["R"],
                pose["t"],
                filename=f"overlay_k{int(k_try):02d}_kabsch.png",
                title=(
                    f"K={k_try} Kabsch  angle={float(pose['angle_deg']):.1f}°  "
                    f"cos={cosine:.3f}  inliers={n_in}/{len(moving_corr)}"
                ),
                mov_centroids_mm=moving_corr,
                fix_centroids_mm=fixed_corr,
            )
            trial_overlays.append({
                "k": int(k_try),
                "R": pose["R"],
                "t": pose["t"],
                "angle_deg": float(pose["angle_deg"]),
                "n_inliers": n_in,
                "n_landmarks": int(len(moving_corr)),
                "cosine": cosine,
                "mov_mask": ov_mov_mask,
                "fix_mask": ov_fix_mask,
                "mov_origin": ov_mov_origin,
                "mov_scales": ov_mov_scales,
                "fix_origin": ov_fix_origin,
                "fix_scales": ov_fix_scales,
                "mov_centroids": moving_corr,
                "fix_centroids": fixed_corr,
            })
        if key > best_key:
            best_key = key
            best_pick = pose
            best_packed = packed
            best_record = {
                **record,
                **pose,
                "k": int(k_try),
                "status": "coarse_ok",
                "cosine": cosine,
            }

    if rank_tensors is not None:
        del rank_tensors
        torch.cuda.empty_cache()

    meta = {
        "trials": trials,
        "nested_fg": bool(nested_fg),
        "inlier_mm": float(inlier_mm),
        "best_k": None if best_pick is None else int(best_record.get("k")),
        "estimator": "weighted_kabsch",
        "rank_metric": "feature_cosine",
        "rank_pool": int(SEMANTIC_K_RANK_POOL),
        "k_tries": list(SEMANTIC_LANDMARK_K_TRIES),
    }
    if debug is not None:
        debug.meta["semantic_landmarks_refine"] = {**meta, **best_record}
        debug.save_k_trials_summary(trials, best_k=meta["best_k"])
        if trial_overlays:
            debug.save_k_trials_grid(trial_overlays, best_k=meta["best_k"])

    if best_pick is None:
        _log(scan_name, "semantic landmarks: no acceptable Kabsch over K tries")
        return None

    k_best = int(best_record["k"])
    label_k = f"k={k_best}"
    R0 = np.asarray(best_pick["R"], dtype=np.float64)
    t0 = np.asarray(best_pick["t"], dtype=np.float64).reshape(3)
    angle0 = float(_rigid_rotation_angle_deg(R0))
    cos0 = float(best_record.get("cosine", float("nan")))
    trial_bits = ", ".join(
        (
            f"K={t['k']}:cos={t.get('cosine', float('nan')):.3f}/ang={t.get('angle_deg', 0):.0f}°"
            if t.get("pose_status") == "ok"
            else f"K={t['k']}:{t.get('pose_status') or t.get('landmark_status')}"
        )
        for t in trials
    )
    _log(
        scan_name,
        f"semantic landmarks coarse: chose {label_k} ({best_pick['label']}) "
        f"cosine={cos0:.3f} angle={angle0:.1f}° inliers={best_pick['n_inliers']} "
        f"med_res={best_pick['median_residual_mm']:.2f} mm | trials: {trial_bits}",
    )
    if debug is not None and best_packed is not None:
        _mc, _fc, _sc, mov_grid, fix_grid, mask_m, mask_f = best_packed
        ov_mov_mask = dbg_mov_mask if dbg_mov_mask is not None else mask_m
        ov_fix_mask = dbg_fix_mask if dbg_fix_mask is not None else mask_f
        ov_mov_origin = dbg_mov_origin if dbg_mov_origin is not None else mov_origin
        ov_fix_origin = dbg_fix_origin if dbg_fix_origin is not None else fix_origin
        ov_mov_scales = dbg_mov_scales if dbg_mov_scales is not None else mov_scales
        ov_fix_scales = dbg_fix_scales if dbg_fix_scales is not None else fix_scales
        debug.save_cluster_label_views(mov_grid, mask_m, k_best, "moving_clusters")
        debug.save_cluster_label_views(fix_grid, mask_f, k_best, "fixed_clusters")
        debug.save_landmark_figure(_mc, _fc, _sc, status=f"cluster CoM {label_k}")
        debug.save_rigid_mesh_overlay(
            ov_mov_mask,
            ov_fix_mask,
            ov_mov_origin,
            ov_mov_scales,
            ov_fix_origin,
            ov_fix_scales,
            R0,
            t0,
            filename="overlay_best_kabsch.png",
            title=f"best {label_k} Kabsch  cos={cos0:.3f}  angle={angle0:.1f}°",
            mov_centroids_mm=_mc,
            fix_centroids_mm=_fc,
        )
        debug.save_winning_cluster_side_by_side(
            mov_grid,
            fix_grid,
            mask_m,
            mask_f,
            mov_origin,
            mov_scales,
            fix_origin,
            fix_scales,
            R0,
            t0,
            k=k_best,
            mov_centroids_mm=_mc,
            fix_centroids_mm=_fc,
            filename="overlay_best_clusters_side_by_side.png",
            title=f"best {label_k} clusters  cos={cos0:.3f}  angle={angle0:.1f}°",
            mov_mask_native=dbg_mov_mask,
            fix_mask_native=dbg_fix_mask,
            mov_origin_native_mm=dbg_mov_origin,
            mov_scales_native_mm=dbg_mov_scales,
            fix_origin_native_mm=dbg_fix_origin,
            fix_scales_native_mm=dbg_fix_scales,
        )
    device = torch.device("cuda", 0)
    aa0 = _rotation_to_axis_angle(R0).astype(np.float32)
    # p_fixed = R @ p_moving + t  ⇒  delta = t - (c_f - R @ c_m) with centroid form
    # p_fixed = R @ (p_moving - c_m) + c_f + delta  ⇒  delta = t - c_f + R @ c_m
    delta0 = (t0 - c_f + R0 @ c_m).astype(np.float32)

    hit1 = None
    tensors_p2 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_COARSE_POOL,
        device,
        mov_intensity=mov_intensity,
        fix_intensity=fix_intensity,
    )
    if tensors_p2 is not None:
        hit1 = _adam_refine_semantic_pose(
            tensors_p2,
            init_axis_angle=aa0,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta0,
            n_steps=SEMANTIC_REFINE_COARSE_STEPS,
            lr=SEMANTIC_REFINE_COARSE_LR,
        )
        aa_r = hit1["axis_angle"].detach().cpu().numpy().astype(np.float32)
        delta_r = hit1["delta"].detach().cpu().numpy().astype(np.float32)
        if debug is not None and best_packed is not None:
            R_c = _axis_angle_to_rotation_matrix(hit1["axis_angle"]).detach().cpu().numpy()
            t_c = c_f + delta_r.astype(np.float64) - R_c @ c_m
            _mc, _fc, _sc, _mg, _fg, mask_m, mask_f = best_packed
            debug.save_rigid_mesh_overlay(
                ov_mov_mask if dbg_mov_mask is not None else mask_m,
                ov_fix_mask if dbg_fix_mask is not None else mask_f,
                ov_mov_origin if dbg_mov_origin is not None else mov_origin,
                ov_mov_scales if dbg_mov_scales is not None else mov_scales,
                ov_fix_origin if dbg_fix_origin is not None else fix_origin,
                ov_fix_scales if dbg_fix_scales is not None else fix_scales,
                R_c,
                t_c,
                filename="overlay_cluster_refine_coarse.png",
                title=(
                    f"refine coarse (pool÷{SEMANTIC_REFINE_COARSE_POOL})  "
                    f"cos={float(hit1['cosine']):.3f}"
                ),
                mov_centroids_mm=_mc,
                fix_centroids_mm=_fc,
            )
        del tensors_p2
        torch.cuda.empty_cache()
    else:
        aa_r, delta_r = aa0, delta0

    tensors_p1 = _prepare_feature_map_volumes(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        SEMANTIC_REFINE_FINE_POOL,
        device,
        mov_intensity=mov_intensity,
        fix_intensity=fix_intensity,
    )
    if tensors_p1 is not None:
        hit2 = _adam_refine_semantic_pose(
            tensors_p1,
            init_axis_angle=aa_r,
            c_m=c_m,
            c_f=c_f,
            init_delta=delta_r,
            n_steps=SEMANTIC_REFINE_FINE_STEPS,
            lr=SEMANTIC_REFINE_FINE_LR,
        )
        final_aa = hit2["axis_angle"]
        final_delta = hit2["delta"]
        final_cos = float(hit2["cosine"])
        final_score = float(hit2["score"])
        del tensors_p1
        torch.cuda.empty_cache()
    elif hit1 is not None:
        final_aa = hit1["axis_angle"]
        final_delta = hit1["delta"]
        final_cos = float(hit1["cosine"])
        final_score = float(hit1["score"])
    else:
        final_aa = torch.as_tensor(aa0, dtype=torch.float32, device=device)
        final_delta = torch.as_tensor(delta0, dtype=torch.float32, device=device)
        final_cos = float("nan")
        final_score = float("nan")

    R = _axis_angle_to_rotation_matrix(final_aa).detach().cpu().numpy().astype(np.float64)
    delta = final_delta.detach().cpu().numpy().astype(np.float64).reshape(3)
    t = c_f + delta - R @ c_m
    angle = float(_rigid_rotation_angle_deg(R))
    probe_name = label_k

    _log(
        scan_name,
        f"semantic landmarks refine: {label_k} cos={final_cos:.3f} "
        f"score={final_score:.3f} angle={angle0:.1f}°→{angle:.1f}°",
    )
    best_record.update({
        "status": "accepted",
        "refined_cosine": final_cos,
        "refined_score": final_score,
        "refined_angle_deg": angle,
        "coarse_angle_deg": angle0,
    })
    if debug is not None:
        debug.meta["semantic_landmarks_refine"] = {**meta, **best_record}
        debug.meta["accepted_pose"] = {
            "R": R.tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "angle_deg": angle,
            "label": label_k,
            "probe_name": probe_name,
            "source": "semantic_landmarks_refine",
            "refined_cosine": final_cos,
        }
        if best_packed is not None:
            _mc, _fc, _sc, _mg, _fg, mask_m, mask_f = best_packed
            debug.save_rigid_mesh_overlay(
                ov_mov_mask if dbg_mov_mask is not None else mask_m,
                ov_fix_mask if dbg_fix_mask is not None else mask_f,
                ov_mov_origin if dbg_mov_origin is not None else mov_origin,
                ov_mov_scales if dbg_mov_scales is not None else mov_scales,
                ov_fix_origin if dbg_fix_origin is not None else fix_origin,
                ov_fix_scales if dbg_fix_scales is not None else fix_scales,
                R,
                t,
                filename="overlay_cluster_refine_fine.png",
                title=(
                    f"refine fine (pool÷{SEMANTIC_REFINE_FINE_POOL})  "
                    f"cos={final_cos:.3f}  angle={angle:.1f}°"
                ),
                mov_centroids_mm=_mc,
                fix_centroids_mm=_fc,
            )
    return _dino_reg_pose(
        R,
        t,
        c_m,
        c_f,
        "semantic_landmarks_refine",
        label=label_k,
        probe_name=probe_name,
        angle_deg=angle,
        refined_cosine=final_cos,
        n_inliers=int(best_pick["n_inliers"]),
    )


def _rigid_from_semantic_landmarks(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    mov_diag_mm,
    fix_diag_mm,
    scan_name=None,
    debug=None,
):
    """
    Correspondence-free rigid from shared PCA-space cluster centroids.

    Joint k-means paints both volumes with the same semantic materials; aligning
    those materials' centres of mass recovers orientation even when the blobs
    do not overlap in mm.
    """
    packed = None
    record = {"status": "not_started"}
    for k_try in SEMANTIC_LANDMARK_K_TRIES:
        packed, record = _semantic_landmarks_from_clusters(
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            k=k_try,
        )
        if packed is not None:
            break

    if debug is not None:
        debug.meta["semantic_landmarks"] = record

    if packed is None:
        _log(
            scan_name,
            f"semantic landmarks: {record.get('status', 'failed')} "
            f"(k={record.get('k')}, landmarks={record.get('n_landmarks', 0)})",
        )
        return None

    moving_corr, fixed_corr, scores, mov_grid, fix_grid, mask_m, mask_f = packed
    nested_fg = _is_nested_foreground(mov_diag_mm, fix_diag_mm)
    inlier_mm = max(4.0, 0.16 * min(float(mov_diag_mm), float(fix_diag_mm)))
    if nested_fg:
        inlier_mm = max(4.0, 0.22 * float(mov_diag_mm))
    min_inliers = max(MIN_LANDMARKS, len(moving_corr) // 2)
    spread_mm = max(1.5, 0.08 * min(float(mov_diag_mm), float(fix_diag_mm)))
    record["nested_fg"] = bool(nested_fg)
    record["inlier_mm"] = float(inlier_mm)
    record["min_inliers"] = int(min_inliers)

    _log(
        scan_name,
        f"semantic landmarks: {len(moving_corr)} shared clusters "
        f"(k={record['k']}, inlier≤{inlier_mm:.1f} mm)",
    )
    if debug is not None:
        debug.save_cluster_label_views(mov_grid, mask_m, record["k"], "moving_clusters")
        debug.save_cluster_label_views(fix_grid, mask_f, record["k"], "fixed_clusters")
        debug.save_landmark_figure(moving_corr, fixed_corr, scores, status="cluster CoM")
        debug.meta["semantic_landmarks"] = record

    pose = _pick_rigid_from_landmark_pairs(
        moving_corr,
        fixed_corr,
        scores,
        c_m,
        c_f,
        inlier_mm,
        spread_mm,
        min_inliers,
        scan_name=scan_name,
        stage_label=" semantic-landmarks",
    )
    if pose is None:
        _log(scan_name, "semantic landmarks: pose rejected (Kabsch / identity guard)")
        record["status"] = "pose_rejected"
        if debug is not None:
            debug.meta["semantic_landmarks"] = record
        return None

    record.update(pose)
    record["status"] = "accepted"
    _log(
        scan_name,
        f"semantic landmarks accepted ({pose['label']}): angle={pose['angle_deg']:.1f}° "
        f"inliers={pose['n_inliers']}/{len(moving_corr)} "
        f"med_res={pose['median_residual_mm']:.2f} mm",
    )
    if debug is not None:
        debug.meta["semantic_landmarks"] = record
        debug.meta["accepted_pose"] = {
            "R": np.asarray(pose["R"], dtype=np.float64).tolist(),
            "t": np.asarray(pose["t"], dtype=np.float64).tolist(),
            "angle_deg": float(pose["angle_deg"]),
            "label": str(pose["label"]),
            "source": "semantic_landmarks",
        }
    return _dino_reg_pose(
        pose["R"],
        pose["t"],
        c_m,
        c_f,
        "semantic_landmarks",
        label=str(pose["label"]),
        angle_deg=float(pose["angle_deg"]),
        n_inliers=int(pose["n_inliers"]),
    )


def _rigid_from_feature_matches(
    mov_feat,
    fix_feat,
    moving_mask_feat,
    fixed_mask_feat,
    mov_scales,
    mov_origin,
    fix_scales,
    fix_origin,
    c_m,
    c_f,
    moving_mask_full,
    fixed_mask_full,
    moving_vs,
    fixed_vs,
    z0_m,
    z0_f,
    scan_name=None,
    debug=None,
):
    """
    Cheap feature pyramid (÷2 → ÷1) on one DINO encode, coarse to fine.

    Each level matches on semantics then estimates rigid via RANSAC; finer levels
    reuse the best pose found so far when scoring candidates.
    Returns synthetic guidepoints for the shared warp, or None.
    """
    mov_diag = _foreground_diagonal_mm(moving_mask_full, moving_vs, z0=z0_m)
    fix_diag = _foreground_diagonal_mm(fixed_mask_full, fixed_vs, z0=z0_f)

    best_overall = None
    best_overall_score = (-1.0, -1, -999)

    for pool in FEATURE_PYRAMID:
        hit = _try_rigid_at_pyramid_level(
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            c_m,
            c_f,
            mov_diag,
            fix_diag,
            pool,
            scan_name=scan_name,
            debug=debug,
        )
        if hit is None:
            continue
        R_lvl, t_lvl, label, score_tuple, _corr = hit
        weighted, n_in, _pool, angle = score_tuple
        pick = (weighted, n_in, -int(_pool), float(angle))
        if pick > best_overall_score:
            best_overall_score = pick
            best_overall = (R_lvl, t_lvl, label, int(_pool), n_in, float(angle))

    if best_overall is None:
        _log(scan_name, "feature pyramid found no acceptable rigid at any scale")
        if debug is not None:
            debug.save_pyramid_summary_chart()
        return None

    R_best, t_best, label, pool, n_in, angle = best_overall
    from .rigidAlignment import _rigid_rotation_angle_deg

    _log(
        scan_name,
        f"accepted rigid ({label}, pyramid ÷{pool}): angle={angle:.1f}° inliers={n_in}",
    )
    if debug is not None:
        debug.save_pyramid_summary_chart()
    if debug is not None:
        debug.meta["accepted_pose"] = {
            "R": np.asarray(R_best, dtype=np.float64).tolist(),
            "t": np.asarray(t_best, dtype=np.float64).tolist(),
            "angle_deg": float(angle),
            "label": str(label),
            "source": "patch_pyramid",
            "pyramid_pool": int(pool),
        }
    return _dino_reg_pose(
        R_best,
        t_best,
        c_m,
        c_f,
        "patch_pyramid",
        label=str(label),
        angle_deg=float(angle),
        n_inliers=int(n_in),
        pyramid_pool=int(pool),
    )


def _release_dinov2_vram(model=None):
    """Drop the singleton ViT after encoding so PCA / matching have room."""
    import gc
    import torch

    from .dinoReg import loader

    held = model if model is not None else loader._MODEL
    if held is not None:
        try:
            held.to("cpu")
        except Exception:
            pass
        del held
    loader._MODEL = None
    loader._DEVICE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(None, "DINOv3 weights released from GPU")


def find_dino_reg_rigid_pose(
    fixed_volume,
    moving_volume,
    fixed_voxel_size,
    moving_voxel_size,
    *,
    scan_name=None,
    reference_name=None,
    moving_threshold=None,
    fixed_threshold=None,
    debug_dir=None,
    progress_callback: ProgressCallback = None,
) -> Optional[dict]:
    """
    Estimate a physical rigid pose from DINOv3 (two-pass).

    Returns a pose dict ``{R, t, scan_centroid_mm, reference_centroid_mm, source, …}``
    with ``p_fixed = R @ p_moving + t`` in mm, or None when no path succeeds.

    Pass 1: cheap octahedral encode (24 proper 90° relabels).
    Pass 2: highest-res tri-planar joint PCA that fits, Adam refine from the
    winning cube. Multi-K cluster CoM Kabsch if the cube is weak or refine
    cosine stays poor. Last resort: patch-NN pyramid.
    Pass ``debug_dir=None`` to skip forensic folder writes.
    """
    debug = None
    if debug_dir:
        from .dinoRegDebug import DinoRegDebugWriter

        debug = DinoRegDebugWriter(
            debug_dir,
            scan_name=scan_name or "scan",
            reference_name=reference_name or "reference",
        )
        _log(scan_name, f"debug artefacts → {debug_dir}")

    try:
        return _find_dino_reg_correspondences_impl(
            fixed_volume,
            moving_volume,
            fixed_voxel_size,
            moving_voxel_size,
            scan_name=scan_name,
            reference_name=reference_name,
            moving_threshold=moving_threshold,
            fixed_threshold=fixed_threshold,
            debug=debug,
            progress_callback=progress_callback,
        )
    finally:
        if debug is not None:
            debug.finalize()


def _find_dino_reg_correspondences_impl(
    fixed_volume,
    moving_volume,
    fixed_voxel_size,
    moving_voxel_size,
    *,
    scan_name=None,
    reference_name=None,
    moving_threshold=None,
    fixed_threshold=None,
    debug=None,
    progress_callback: ProgressCallback = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    from .rigidAlignment import _rigid_rotation_angle_deg

    ensure_dino_reg_cuda()
    _notify(progress_callback, f"Loading DINOv3 for {scan_name or 'scan'}...")
    model, device = get_dino_model()

    fixed_vs = float(fixed_voxel_size)
    moving_vs = float(moving_voxel_size)

    _notify(progress_callback, f"Normalizing volumes for DINO-Reg ({scan_name or 'scan'})...")
    fixed_norm = _normalize_volume(fixed_volume)
    moving_norm = _normalize_volume(moving_volume)
    fixed_mask = _foreground_mask_from_raw(fixed_volume, fixed_threshold)
    moving_mask = _foreground_mask_from_raw(moving_volume, moving_threshold)
    if fixed_mask.sum() < 64 or moving_mask.sum() < 64:
        _log(scan_name, "too little foreground for DINO-Reg")
        if debug is not None:
            debug.set_outcome(False, "too little foreground")
        return None

    z0_m, z1_m = _foreground_z_extent(moving_mask)
    z0_f, z1_f = _foreground_z_extent(fixed_mask)
    moving_crop = moving_norm[:, :, z0_m:z1_m]
    fixed_crop = fixed_norm[:, :, z0_f:z1_f]
    moving_mask_c = moving_mask[:, :, z0_m:z1_m]
    fixed_mask_c = fixed_mask[:, :, z0_f:z1_f]

    c_m = _mask_centroid_mm(moving_mask, moving_vs, z0=0)
    c_f = _mask_centroid_mm(fixed_mask, fixed_vs, z0=0)
    mov_diag = _foreground_diagonal_mm(moving_mask, moving_vs, z0=z0_m)
    fix_diag = _foreground_diagonal_mm(fixed_mask, fixed_vs, z0=z0_f)

    _notify(
        progress_callback,
        f"Octahedral orientation search for {scan_name or 'scan'}...",
    )
    oct_hit = _octahedral_cheap_encode_search(
        moving_norm,
        fixed_norm,
        moving_mask,
        fixed_mask,
        c_m,
        c_f,
        model,
        device,
        scan_name=scan_name,
        debug=debug,
    )

    if oct_hit is not None:
        R_cube = np.asarray(oct_hit["R"], dtype=np.float64)
        cube_name = str(oct_hit["name"])
        cube_angle = float(oct_hit["angle_deg"])
        weak_cube = bool(oct_hit.get("weak", False))
    else:
        R_cube = np.eye(3, dtype=np.float64)
        cube_name = "identity"
        cube_angle = 0.0
        weak_cube = True

    t_cube = _centroid_locked_t(R_cube, c_m, c_f)

    # Pre-rotate moving volume by the winning octahedral relabel so moving and
    # fixed share canonical anatomical slice orientations during high-res encode.
    moving_norm_p2 = _apply_octahedral_relabel(moving_norm, R_cube)
    moving_mask_p2 = _apply_octahedral_relabel(moving_mask, R_cube)

    z0_m_p2, z1_m_p2 = _foreground_z_extent(moving_mask_p2)
    moving_crop_p2 = moving_norm_p2[:, :, z0_m_p2:z1_m_p2]
    moving_mask_c_p2 = moving_mask_p2[:, :, z0_m_p2:z1_m_p2]
    c_m_p2 = _mask_centroid_mm(moving_mask_p2, moving_vs, z0=0)
    mov_diag_p2 = _foreground_diagonal_mm(moving_mask_p2, moving_vs, z0=z0_m_p2)

    overlay_geom_orig = _debug_native_overlay_geom(
        moving_mask_c, fixed_mask_c, moving_vs, fixed_vs, z0_m, z0_f
    )
    if debug is not None and overlay_geom_orig[0] is not None:
        debug.save_rigid_mesh_overlay(
            overlay_geom_orig[0], overlay_geom_orig[1],
            overlay_geom_orig[2], overlay_geom_orig[3],
            overlay_geom_orig[4], overlay_geom_orig[5],
            R_cube, t_cube,
            filename="overlay_octahedral_pass1.png",
            title=(
                f"octahedral {cube_name}  "
                f"pass1={float(oct_hit['cosine'] if oct_hit else 0.0):.3f}  "
                f"ang={cube_angle:.1f}°"
            ),
        )

    import torch

    torch.cuda.empty_cache()

    mov_feat, fix_feat, feat_h, feat_w = _encode_pair_highest_res(
        moving_crop_p2,
        fixed_crop,
        moving_mask_c_p2,
        fixed_mask_c,
        model,
        device,
        scan_name=scan_name,
        progress_callback=progress_callback,
    )
    _release_dinov2_vram(model)
    model = None

    moving_mask_feat = _resize_mask_to_feat(moving_mask_c_p2, feat_h, feat_w)
    fixed_mask_feat = _resize_mask_to_feat(fixed_mask_c, feat_h, feat_w)
    if mov_feat is None or fix_feat is None:
        _log(scan_name, "joint PCA / streaming encode failed")
        if debug is not None:
            debug.set_outcome(False, "streaming encode/PCA failed")
        return None
    _log(
        scan_name,
        f"streamed joint PCA k={REG_FEATURE_DIM} "
        f"mov={list(mov_feat.shape)} fix={list(fix_feat.shape)} "
        f"mov_fg={int(moving_mask_feat.sum())} fix_fg={int(fixed_mask_feat.sum())}",
    )

    mov_scales, mov_origin = _feat_scales_mm(moving_crop_p2.shape, mov_feat.shape[:3], moving_vs, z0_m_p2)
    fix_scales, fix_origin = _feat_scales_mm(fixed_crop.shape, fix_feat.shape[:3], fixed_vs, z0_f)

    overlay_geom_p2 = _debug_native_overlay_geom(
        moving_mask_c_p2, fixed_mask_c, moving_vs, fixed_vs, z0_m_p2, z0_f
    )
    if overlay_geom_p2[0] is None:
        overlay_geom_p2 = (
            moving_mask_feat, fixed_mask_feat,
            mov_origin, mov_scales, fix_origin, fix_scales,
        )

    if debug is not None:
        debug.set_run_context(
            feat_grid=[int(feat_h), int(feat_w)],
            slice_gap=SLICE_GAP,
            pca_dim=REG_FEATURE_DIM,
            moving_shape=list(moving_crop_p2.shape),
            fixed_shape=list(fixed_crop.shape),
            moving_voxel_size=moving_vs,
            fixed_voxel_size=fixed_vs,
            moving_centroid_mm=c_m.tolist(),
            fixed_centroid_mm=c_f.tolist(),
            moving_fg_diag_mm=float(mov_diag_p2),
            fixed_fg_diag_mm=float(fix_diag),
            nested_fg=bool(_is_nested_foreground(mov_diag_p2, fix_diag)),
            pyramid=list(FEATURE_PYRAMID),
            encoding="octahedral_pass1_prerotate_triplanar_pca",
            octahedral_pass1=(oct_hit["record"] if oct_hit else None),
        )
        debug.save_pca_semantic_views(
            mov_feat, moving_mask_feat, "moving_full",
            title_prefix=f"{scan_name or 'moving'} moving (pre-aligned {cube_name})",
            scales_mm=mov_scales,
        )
        debug.save_pca_semantic_views(
            fix_feat, fixed_mask_feat, "fixed_full",
            title_prefix=f"{reference_name or 'fixed'} fixed",
            scales_mm=fix_scales,
        )

    # High-res pre-aligned baseline: identity
    hi_cos_id = _score_feature_cosine_at_R(
        mov_feat, fix_feat, moving_mask_feat, fixed_mask_feat,
        mov_scales, mov_origin, fix_scales, fix_origin,
        np.eye(3), c_m_p2, c_f,
    )
    _log(
        scan_name,
        f"pre-aligned {cube_name} high-res identity cosine={hi_cos_id:.3f} "
        f"(pass-1 cos={float(oct_hit['cosine'] if oct_hit else 0.0):.3f})",
    )

    # Residual refinement: constrained cluster Kabsch + Adam refine
    _notify(
        progress_callback,
        f"Semantic landmarks & cosine refine for {scan_name or 'scan'}...",
    )
    cluster_res = _rigid_from_semantic_landmarks_and_refine(
        mov_feat,
        fix_feat,
        moving_mask_feat,
        fixed_mask_feat,
        mov_scales,
        mov_origin,
        fix_scales,
        fix_origin,
        c_m_p2,
        c_f,
        mov_diag_p2,
        fix_diag,
        scan_name=scan_name,
        debug=debug,
        mov_intensity=moving_crop_p2,
        fix_intensity=fixed_crop,
        mov_mask_native=moving_mask_c_p2,
        fix_mask_native=fixed_mask_c,
        mov_vs=moving_vs,
        fix_vs=fixed_vs,
        z0_m=z0_m_p2,
        z0_f=z0_f,
        max_residual_angle_deg=35.0,
    )

    if cluster_res is not None:
        R_res = np.asarray(cluster_res["R"], dtype=np.float64)
        t_res = np.asarray(cluster_res["t"], dtype=np.float64)
        final_cos = _pose_refined_cosine(cluster_res)
        source_label = cluster_res.get("source", "semantic_landmarks_refine")
    else:
        # Fallback to direct Adam refine from identity
        R_res, t_res, final_cos, final_score, _, _ = _adam_refine_from_coarse_rt(
            np.eye(3, dtype=np.float64),
            c_f - c_m_p2,
            c_m_p2,
            c_f,
            mov_feat,
            fix_feat,
            moving_mask_feat,
            fixed_mask_feat,
            mov_scales,
            mov_origin,
            fix_scales,
            fix_origin,
            mov_intensity=moving_crop_p2,
            fix_intensity=fixed_crop,
            scan_name=scan_name,
            debug=debug,
            overlay_geom=overlay_geom_p2,
            source_label=f"prealigned identity ({cube_name})",
            overlay_prefix="octahedral_",
        )
        source_label = "prealigned_identity_refine"

    # Compose total transform: p_fixed = R_total @ p_moving_orig + t_total
    R_total = R_res @ R_cube
    delta_t = t_res - c_f + R_res @ c_m_p2
    t_total = c_f - R_total @ c_m + delta_t
    ang_total = float(_rigid_rotation_angle_deg(R_total))
    ang_res = float(_rigid_rotation_angle_deg(R_res))

    result = _dino_reg_pose(
        R_total,
        t_total,
        c_m,
        c_f,
        "octahedral_prealigned_refine",
        label=f"{cube_name}_refined",
        probe_name=cube_name,
        angle_deg=ang_total,
        residual_angle_deg=ang_res,
        refined_cosine=final_cos,
        pass1_cosine=(float(oct_hit["cosine"]) if oct_hit else None),
        highres_seed_cosine=(float(hi_cos_id) if np.isfinite(hi_cos_id) else None),
        weak_pass1=weak_cube,
    )

    _log(
        scan_name,
        f"accepted {cube_name}: total angle={ang_total:.1f}° (residual={ang_res:.1f}°) "
        f"refined_cosine={final_cos:.3f}",
    )

    if debug is not None and result is not None:
        debug.meta["accepted_pose"] = {
            "R": np.asarray(result["R"], dtype=np.float64).tolist(),
            "t": np.asarray(result["t"], dtype=np.float64).tolist(),
            "angle_deg": float(result.get("angle_deg", 0.0)),
            "residual_angle_deg": float(result.get("residual_angle_deg", 0.0)),
            "label": str(result.get("label", result.get("source"))),
            "source": str(result.get("source")),
            "refined_cosine": result.get("refined_cosine"),
        }
        if overlay_geom_orig[0] is not None:
            debug.save_rigid_mesh_overlay(
                overlay_geom_orig[0],
                overlay_geom_orig[1],
                overlay_geom_orig[2],
                overlay_geom_orig[3],
                overlay_geom_orig[4],
                overlay_geom_orig[5],
                result["R"],
                result["t"],
                filename="overlay_accepted.png",
                title=(
                    f"accepted {cube_name}  "
                    f"cos={final_cos:.3f}  "
                    f"ang={ang_total:.1f}° (res={ang_res:.1f}°)"
                ),
            )
        debug.set_outcome(
            True,
            "accepted rigid",
            {
                **result,
                **(debug.meta.get("accepted_pose") or {}),
            },
        )
    return result


def find_dino_reg_correspondences(*args, **kwargs):
    """Deprecated alias; use :func:`find_dino_reg_rigid_pose`."""
    return find_dino_reg_rigid_pose(*args, **kwargs)
