"""
GPU Rigid: FireANTs on the union-canvas estimation grid, then full-canvas apply.

Pipeline:
  1. Centroid paste upstream (identity baseline on estimation grid).
  2. MomentsRegistration at the coarsest pyramid level (intensity-only) → rigid init.
  3. RigidRegistration on finer pyramid levels (masked when enabled) starting from that init.
  4. Select the best pose among {identity, moments, refine} by a fixed overlap metric order.

Selection is data-driven on every pair (no per-subject tuning). Moments never sees mask
channels; refine may use masked similarity when foreground masks are available.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

FIREANTS_MIN_AXIS_VOXELS = 32
DEFAULT_GPU_RIGID_SCALES = (8, 4, 2, 1)
DEFAULT_GPU_RIGID_ITERATIONS = (80, 100, 150, 180)
DEFAULT_GPU_RIGID_TRANSL_MODE = "cof"
os.environ.setdefault("USE_FFO", "false")

_FIREANTS_LOGGERS = (
    "fireants",
    "fireants.registration",
    "fireants.registration.abstract",
    "fireants.registration.moments",
    "fireants.registration.rigid",
    "fireants.losses",
    "fireants.losses.cc",
    "fireants.utils",
    "fireants.utils.imageutils",
    "fireants.interpolator",
    "fireants.io",
)
_FUSED_OPS_NOTE_EMITTED = False


@contextlib.contextmanager
def _quiet_fireants_library_output(scan_name=None) -> Iterator[None]:
    global _FUSED_OPS_NOTE_EMITTED
    if not _FUSED_OPS_NOTE_EMITTED:
        _gpu_rigid_log(
            scan_name,
            "FireANTs fused_ops not installed (expected for v1); "
            "using PyTorch FFT and standard downsampling",
        )
        _FUSED_OPS_NOTE_EMITTED = True
    prev_levels: dict[str, int] = {}
    for name in _FIREANTS_LOGGERS:
        logger = logging.getLogger(name)
        prev_levels[name] = logger.level
        logger.setLevel(logging.ERROR)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)


def _rigid_alignment():
    from . import rigidAlignment

    return rigidAlignment


class GpuRigidError(RuntimeError):
    """Raised when GPU Rigid (FireANTs) cannot run."""


ProgressCallback = Optional[Callable[[str], None]]


def ensure_gpu_rigid_cuda():
    try:
        import torch
    except ImportError as exc:
        raise GpuRigidError(
            "GPU Rigid requires PyTorch with CUDA. PyTorch is not installed."
        ) from exc
    if not torch.cuda.is_available():
        raise GpuRigidError(
            "GPU Rigid requires a CUDA GPU. No CUDA device is available."
        )
    try:
        import fireants  # noqa: F401
    except ImportError as exc:
        raise GpuRigidError(
            "GPU Rigid requires the fireants package. Install with: pip install fireants==1.5.0"
        ) from exc


def is_ants_style_physical_rigid_method(method) -> bool:
    return str(method or "").strip().lower() in ("ants", "gpu-rigid")


def normalize_gpu_rigid_options(request_data):
    raw = None
    if isinstance(request_data, dict):
        raw = request_data.get("gpu_rigid_options")
    if not isinstance(raw, dict):
        raw = {}

    def _clip_int(v, lo, hi, default):
        try:
            return max(lo, min(hi, int(v)))
        except (TypeError, ValueError):
            return default

    def _clip_float(v, lo, hi, default):
        try:
            fv = float(v)
            if fv != fv:
                return default
            return max(lo, min(hi, fv))
        except (TypeError, ValueError):
            return default

    def _clip_int_tuple(seq, lo, hi, defaults):
        if not isinstance(seq, (list, tuple)) or len(seq) != len(defaults):
            return tuple(defaults)
        return tuple(_clip_int(x, lo, hi, d) for x, d in zip(seq, defaults))

    initializer = str(raw.get("initializer", "moments")).strip().lower()
    if initializer not in {"moments", "identity"}:
        initializer = "moments"

    transl_mode = str(raw.get("transl_mode", DEFAULT_GPU_RIGID_TRANSL_MODE)).strip().lower()
    if transl_mode not in {"com", "cof"}:
        transl_mode = DEFAULT_GPU_RIGID_TRANSL_MODE

    allowed_losses = {"auto", "cc", "mi", "mse", "ngf", "nmi", "ngf_nmi"}
    loss_type = str(raw.get("loss_type", "auto")).strip().lower()
    if loss_type not in allowed_losses:
        loss_type = "auto"

    optimizer = str(raw.get("optimizer", "Adam")).strip()
    if optimizer.lower() not in ("adam", "sgd"):
        optimizer = "Adam"

    return {
        "initializer": initializer,
        "transl_mode": transl_mode,
        "loss_type": loss_type,
        "scales": _clip_int_tuple(raw.get("scales"), 1, 64, DEFAULT_GPU_RIGID_SCALES),
        "iterations": _clip_int_tuple(
            raw.get("iterations"), 1, 5000, DEFAULT_GPU_RIGID_ITERATIONS
        ),
        "optimizer": optimizer,
        "optimizer_lr": _clip_float(raw.get("optimizer_lr"), 1e-6, 1.0, 3e-3),
        "cc_kernel_size": _clip_int(raw.get("cc_kernel_size"), 3, 9, 5),
        "use_registration_mask": bool(raw.get("use_registration_mask", True)),
        "refine": bool(raw.get("refine", True)),
        "max_estimation_voxels": _clip_int(
            raw.get("max_estimation_voxels"), 1_000_000, 200_000_000, 40_000_000
        ),
        "adapt_scales": bool(raw.get("adapt_scales", True)),
        "singleprecision": bool(raw.get("singleprecision", True)),
        "debug_checkpoints": bool(raw.get("debug_checkpoints", True)),
    }


def _metrics_rank(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    """Lexicographic objective: reference coverage, overlap quality, intensity agreement."""
    return (
        float(metrics.get("fixed_recall", 0.0)),
        float(metrics.get("iou", 0.0)),
        float(metrics.get("ncc", -1.0)),
    )


def _metrics_better_than(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return _metrics_rank(a) > _metrics_rank(b)


def matrix_to_moving_to_fixed_rt(matrix_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(matrix_4x4, dtype=np.float64)
    if mat.shape == (3, 4):
        return mat[:, :3].copy(), mat[:, 3].copy()
    return mat[:3, :3].copy(), mat[:3, 3].copy()


def disambiguate_moving_to_fixed_rt(
    R_raw: np.ndarray,
    t_raw: np.ndarray,
    fixed_image,
    moving_image,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], str]:
    """
    Project to SO(3) and pick the better of forward vs inverse on the estimation grid.
    """
    ra = _rigid_alignment()
    R_fwd = ra._nearest_rotation_from_linear(R_raw)
    t_fwd = np.asarray(t_raw, dtype=np.float64).reshape(3)
    metrics_fwd = ra._score_rigid_metrics(fixed_image, moving_image, R_fwd, t_fwd)

    R_inv = R_fwd.T
    t_inv = -R_fwd.T @ t_fwd
    metrics_inv = ra._score_rigid_metrics(fixed_image, moving_image, R_inv, t_inv)

    if _metrics_better_than(metrics_inv, metrics_fwd):
        return R_inv, t_inv, metrics_inv, "inverse"
    return R_fwd, t_fwd, metrics_fwd, "forward"


def fireants_level_shape(estimation_shape: Sequence[int], scale: int) -> Tuple[int, int, int]:
    scale = max(1, int(scale))
    return tuple(max(int(s) // scale, FIREANTS_MIN_AXIS_VOXELS) for s in estimation_shape)


def choose_gpu_rigid_scales(
    estimation_shape: Sequence[int],
    preferred_scales: Sequence[int] = DEFAULT_GPU_RIGID_SCALES,
    min_axis_voxels: int = FIREANTS_MIN_AXIS_VOXELS,
    max_levels: int = 4,
) -> Tuple[int, ...]:
    est = tuple(int(s) for s in estimation_shape)
    if not est or min(est) < 1:
        return tuple(int(s) for s in preferred_scales)
    min_dim = min(est)
    floor = max(1, int(min_axis_voxels))
    target_sizes = []
    sz = max(floor, min_dim // 8)
    while sz < min_dim and len(target_sizes) < max_levels - 1:
        target_sizes.append(sz)
        nxt = min(min_dim, max(sz + 1, sz * 2))
        if nxt <= sz:
            break
        sz = nxt
    derived = []
    seen = set()
    for target in reversed(target_sizes):
        scale = max(1, int(round(min_dim / max(target, 1))))
        level = fireants_level_shape(est, scale)
        if level not in seen:
            derived.append(scale)
            seen.add(level)
    if min_dim // max(1, floor) > 1:
        for scale in (2, 1):
            level = fireants_level_shape(est, scale)
            if level not in seen:
                derived.append(scale)
                seen.add(level)
                break
    if not derived:
        derived = [max(1, int(s)) for s in preferred_scales]
    derived = sorted(set(derived), reverse=True)
    filtered = []
    seen = set()
    for scale in derived:
        level = fireants_level_shape(est, scale)
        if level in seen:
            continue
        filtered.append(scale)
        seen.add(level)
    if len(filtered) < 2:
        for scale in sorted(preferred_scales, reverse=True):
            level = fireants_level_shape(est, scale)
            if level not in seen:
                filtered.append(scale)
                seen.add(level)
            if len(filtered) >= 2:
                break
        filtered = sorted(set(filtered), reverse=True)
    return tuple(filtered[:max_levels])


def _match_iterations_to_scales(
    scales: Sequence[int],
    iterations: Sequence[int],
    default_iterations: Sequence[int] = DEFAULT_GPU_RIGID_ITERATIONS,
) -> Tuple[int, ...]:
    scales = tuple(int(s) for s in scales)
    defaults = tuple(int(x) for x in default_iterations)
    iters = tuple(int(x) for x in (iterations or defaults))
    n = len(scales)
    if n == 0:
        return tuple()
    if len(iters) == n:
        return iters
    if n == 1:
        return (iters[-1] if iters else defaults[-1],)
    idx = [round(i * (len(defaults) - 1) / (n - 1)) for i in range(n)]
    return tuple(iters[i] if i < len(iters) else defaults[i] for i in idx)


def format_gpu_rigid_resolution_chain(
    reference_shape: Optional[Sequence[int]],
    canvas_shape: Optional[Sequence[int]],
    estimation_stride: int,
    estimation_shape: Sequence[int],
    scales: Sequence[int],
) -> str:
    est = tuple(int(s) for s in estimation_shape)
    parts = []
    if reference_shape is not None:
        parts.append(f"reference={tuple(int(s) for s in reference_shape)}")
    if canvas_shape is not None:
        parts.append(f"canvas={tuple(int(s) for s in canvas_shape)}")
    parts.append(f"estimation(stride={max(1, int(estimation_stride))})={est}")
    bits = []
    for scale in scales:
        level = fireants_level_shape(est, scale)
        bits.append(f"s{scale}→{level}({int(np.prod(level)):,}vox)")
    parts.append("FireANTs[" + ", ".join(bits) + "]")
    return "; ".join(parts)


def adapt_gpu_rigid_opts_to_grid(
    opts: dict,
    estimation_shape: Sequence[int],
    reference_shape: Optional[Sequence[int]] = None,
    canvas_shape: Optional[Sequence[int]] = None,
    estimation_stride: int = 1,
    scan_name=None,
) -> dict:
    adapted = dict(opts)
    if not adapted.get("adapt_scales", True):
        return adapted
    preferred = tuple(int(s) for s in adapted.get("scales", DEFAULT_GPU_RIGID_SCALES))
    scales = choose_gpu_rigid_scales(estimation_shape, preferred_scales=preferred)
    adapted["scales"] = scales
    adapted["iterations"] = _match_iterations_to_scales(scales, adapted.get("iterations", ()))
    if scales != preferred:
        _gpu_rigid_log(
            scan_name,
            f"adapted scales {list(preferred)} → {list(scales)} for est grid "
            f"{tuple(int(s) for s in estimation_shape)}",
        )
    _gpu_rigid_log(
        scan_name,
        format_gpu_rigid_resolution_chain(
            reference_shape, canvas_shape, estimation_stride, estimation_shape, scales
        ),
    )
    return adapted


def project_homogeneous_affine_to_rigid(matrix_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = _rigid_alignment()
    mat = np.asarray(matrix_4x4, dtype=np.float64)
    if mat.shape == (3, 4):
        A, t = mat[:, :3], mat[:, 3]
    else:
        A, t = mat[:3, :3], mat[:3, 3]
    R = ra._nearest_rotation_from_linear(A)
    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = R
    rigid[:3, 3] = t
    return rigid, R, t


def _rotation_degrees_from_matrix(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _ants_to_fireants_image(ants_image, device: str = "cuda", shift_minus1_to_unit: bool = False):
    import ants
    from fireants.io.image import Image

    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        ants.image_write(ants_image, path)
        image = Image.load_file(path, device=device)
        if shift_minus1_to_unit:
            image.array.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        return image
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _attach_fireants_mask(intensity_image, mask_ants=None, device: str = "cuda"):
    from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones

    if mask_ants is not None:
        mask_image = _ants_to_fireants_image(mask_ants, device=device, shift_minus1_to_unit=False)
        mask_image.array.clamp_(0.0, 1.0)
    else:
        mask_image = generate_image_mask_allones(intensity_image)
        mask_image.array = (intensity_image.array > 0.02).to(mask_image.array.dtype)
    return apply_mask_to_image(intensity_image, mask_image, optimize_memory=True)


def _resolve_refine_loss(
    opts: dict,
    identity_metrics: Dict[str, float],
) -> Tuple[str, Optional[object], str]:
    """Pick FireANTs loss for RigidRegistration from options and identity baseline."""
    from .gpuRigidLosses import build_gpu_rigid_custom_loss

    requested = str(opts.get("loss_type", "auto")).lower()
    if requested == "auto":
        ref_ncc = float(identity_metrics.get("ref_ncc", identity_metrics.get("ncc", -1.0)))
        resolved = "ngf" if ref_ncc < 0.05 else "cc"
    else:
        resolved = requested

    custom = build_gpu_rigid_custom_loss(resolved)
    if custom is not None:
        return "custom", custom, resolved
    if resolved not in {"cc", "mi", "mse"}:
        resolved = "cc"
    return resolved, None, resolved


def _split_moments_refine_pyramid(
    scales: Sequence[int],
    iterations: Sequence[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    scales = [int(s) for s in scales]
    iterations = [int(i) for i in iterations]
    n = min(len(scales), len(iterations))
    scales, iterations = scales[:n], iterations[:n]
    if n <= 1:
        return scales[:1], iterations[:1], [], []
    return scales[:1], iterations[:1], scales[1:], iterations[1:]


def _run_moments_init(
    fixed_fa,
    moving_fa,
    moments_scale: int,
    transl_mode: str,
    scan_name=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    import torch
    from fireants.io.image import BatchedImages
    from fireants.registration.moments import MomentsRegistration

    with _quiet_fireants_library_output(scan_name):
        moments_reg = MomentsRegistration(
            moments_scale,
            BatchedImages([fixed_fa]),
            BatchedImages([moving_fa]),
            moments=2,
            orientation="both",
            transl_mode=transl_mode,
            perform_scaling=False,
        )
        moments_reg.optimize()
    init_dict = moments_reg.get_rigid_init_dict()
    init_moment = init_dict.get("init_moment")
    init_translation = init_dict.get("init_translation")
    if init_moment is None:
        raise GpuRigidError("MomentsRegistration did not return a rotation")
    R = init_moment[0].detach().cpu().numpy()
    t = (
        init_translation[0].detach().cpu().numpy()
        if init_translation is not None
        else np.zeros(3, dtype=np.float64)
    )
    moment_tensor = init_moment.detach()
    transl_tensor = (
        init_translation.detach() if init_translation is not None else None
    )
    del moments_reg
    torch.cuda.empty_cache()
    return R, t, moment_tensor, transl_tensor


def _run_rigid_refine(
    fixed_batch,
    moving_batch,
    scales: Sequence[int],
    iterations: Sequence[int],
    opts: dict,
    identity_metrics: Dict[str, float],
    init_moment,
    init_translation,
    scan_name=None,
):
    import torch
    from fireants.registration.rigid import RigidRegistration

    if not scales:
        return None, None

    dtype = torch.float32 if opts.get("singleprecision", True) else torch.float64
    optimizer = opts["optimizer"]
    if str(optimizer).lower() == "levenberg":
        _gpu_rigid_log(scan_name, "RigidRegistration has no Levenberg optimizer; using Adam")
        optimizer = "Adam"

    loss_type, custom_loss, loss_label = _resolve_refine_loss(opts, identity_metrics)
    use_mask = bool(opts.get("use_registration_mask", True))
    if use_mask and custom_loss is None and not str(loss_type).startswith("masked_"):
        loss_type = f"masked_{loss_type}"

    kwargs = dict(
        scales=list(int(s) for s in scales),
        iterations=list(int(i) for i in iterations),
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type=loss_type,
        optimizer=optimizer,
        optimizer_lr=float(opts["optimizer_lr"]),
        cc_kernel_size=int(opts["cc_kernel_size"]),
        scaling=False,
        progress_bar=False,
        dtype=dtype,
        max_tolerance_iters=25,
        init_moment=init_moment,
    )
    if init_translation is not None:
        kwargs["init_translation"] = init_translation
    if custom_loss is not None:
        kwargs["custom_loss"] = custom_loss

    with _quiet_fireants_library_output(scan_name):
        reg = RigidRegistration(**kwargs)
        reg.optimize()
    matrix = reg.get_rigid_matrix(homogenous=True)[0].detach().cpu().numpy()
    del reg
    torch.cuda.empty_cache()
    R, t = matrix_to_moving_to_fixed_rt(matrix)
    return R, t, loss_label


def _record_debug(
    debug_writer,
    debug_dir,
    scan_name,
    fixed,
    moving,
    R,
    t,
    stage,
    metrics,
    notes="",
    direction: str = "",
):
    if debug_writer is None:
        return
    png = debug_writer.record_checkpoint(
        fixed,
        moving,
        R,
        t,
        stage=stage,
        initializer="fireants",
        metrics=metrics,
        notes=notes,
        direction=direction,
    )
    _gpu_rigid_log(scan_name, f"debug checkpoint → {debug_dir}/{png}")


def _log_pose(scan_name, stage: str, metrics: Dict[str, float], R: np.ndarray, direction: str = ""):
    dir_note = f" dir={direction}" if direction else ""
    _gpu_rigid_log(
        scan_name,
        f"{stage}{dir_note}: recall={metrics['fixed_recall']:.3f} iou={metrics['iou']:.3f} "
        f"ncc={metrics['ncc']:.3f} ref_ncc={metrics.get('ref_ncc', 0):.3f} "
        f"rot≈{_rotation_degrees_from_matrix(R):.1f}°",
    )


def run_gpu_rigid_registration(
    fixed_image,
    moving_image,
    opts: dict,
    scan_name=None,
    progress_callback: ProgressCallback = None,
    fixed_mask=None,
    moving_mask=None,
    reference_shape: Optional[Sequence[int]] = None,
    canvas_shape: Optional[Sequence[int]] = None,
    estimation_stride: int = 1,
    debug_dir: Optional[str] = None,
    reference_name: Optional[str] = None,
):
    """Moments init + optional multi-scale rigid refine; return best-scoring R,t."""
    import torch
    from fireants.io.image import BatchedImages

    ensure_gpu_rigid_cuda()
    ra = _rigid_alignment()
    estimation_shape = tuple(int(s) for s in fixed_image.shape)
    opts = adapt_gpu_rigid_opts_to_grid(
        opts,
        estimation_shape,
        reference_shape=reference_shape,
        canvas_shape=canvas_shape,
        estimation_stride=estimation_stride,
        scan_name=scan_name,
    )

    def _notify(msg: str):
        if progress_callback:
            progress_callback(msg)
        _gpu_rigid_log(scan_name, msg)

    identity_metrics = ra._score_rigid_metrics(fixed_image, moving_image, np.eye(3), np.zeros(3))
    _log_pose(scan_name, "identity baseline", identity_metrics, np.eye(3))

    debug_writer = None
    if opts.get("debug_checkpoints", True) and debug_dir:
        from .gpuRigidDebug import GpuRigidDebugWriter

        debug_writer = GpuRigidDebugWriter(
            debug_dir, scan_name=scan_name or "", reference_name=reference_name or ""
        )
        debug_writer.set_run_context(
            estimation_shape=estimation_shape,
            scales=list(opts["scales"]),
            identity_metrics=identity_metrics,
        )
        _record_debug(
            debug_writer,
            debug_dir,
            scan_name,
            fixed_image,
            moving_image,
            np.eye(3),
            np.zeros(3),
            "identity",
            identity_metrics,
            notes="paste-aligned, R=I",
        )

    device = "cuda"
    transl_mode = str(opts.get("transl_mode", DEFAULT_GPU_RIGID_TRANSL_MODE)).lower()
    moments_scales, moments_iters, refine_scales, refine_iters = _split_moments_refine_pyramid(
        opts["scales"], opts["iterations"]
    )
    moments_scale = int(moments_scales[0]) if moments_scales else 1

    candidates: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, float], str]] = [
        ("identity", np.eye(3), np.zeros(3), identity_metrics, ""),
    ]

    try:
        fixed_fa = _ants_to_fireants_image(fixed_image, device=device, shift_minus1_to_unit=True)
        moving_fa = _ants_to_fireants_image(moving_image, device=device, shift_minus1_to_unit=True)

        init_moment_tensor = None
        init_translation_tensor = None

        if opts.get("initializer", "moments") != "identity":
            _notify(
                f"FireANTs moments at scale={moments_scale} "
                f"(intensity-only, transl_mode={transl_mode})"
            )
            R_raw, t_raw, init_moment_tensor, init_translation_tensor = _run_moments_init(
                fixed_fa,
                moving_fa,
                moments_scale,
                transl_mode,
                scan_name=scan_name,
            )
            R, t, moments_metrics, direction = disambiguate_moving_to_fixed_rt(
                R_raw, t_raw, fixed_image, moving_image
            )
            _log_pose(scan_name, "moments", moments_metrics, R, direction)
            _record_debug(
                debug_writer,
                debug_dir,
                scan_name,
                fixed_image,
                moving_image,
                R,
                t,
                "moments",
                moments_metrics,
                direction=direction,
            )
            candidates.append(("moments", R, t, moments_metrics, direction))

            init_moment_tensor = torch.tensor(
                R, dtype=torch.float32, device=device
            ).unsqueeze(0)
            init_translation_tensor = torch.tensor(
                t, dtype=torch.float32, device=device
            ).unsqueeze(0)

        if opts.get("refine", True) and refine_scales and init_moment_tensor is not None:
            use_mask = bool(opts.get("use_registration_mask", True))
            fixed_refine = fixed_fa
            moving_refine = moving_fa
            if use_mask:
                fixed_refine = _attach_fireants_mask(fixed_refine, fixed_mask, device=device)
                moving_refine = _attach_fireants_mask(moving_refine, moving_mask, device=device)

            _, _, loss_label = _resolve_refine_loss(opts, identity_metrics)
            _notify(
                f"FireANTs rigid refine scales={list(refine_scales)} loss={loss_label} "
                f"mask={'on' if use_mask else 'off'}"
            )
            try:
                R_raw, t_raw, _ = _run_rigid_refine(
                    BatchedImages([fixed_refine]),
                    BatchedImages([moving_refine]),
                    refine_scales,
                    refine_iters,
                    opts,
                    identity_metrics,
                    init_moment_tensor,
                    init_translation_tensor,
                    scan_name=scan_name,
                )
                if R_raw is not None:
                    R, t, refine_metrics, direction = disambiguate_moving_to_fixed_rt(
                        R_raw, t_raw, fixed_image, moving_image
                    )
                    init_metrics = candidates[-1][3]
                    if _metrics_better_than(refine_metrics, init_metrics):
                        _log_pose(scan_name, "refine", refine_metrics, R, direction)
                        _record_debug(
                            debug_writer,
                            debug_dir,
                            scan_name,
                            fixed_image,
                            moving_image,
                            R,
                            t,
                            "refine",
                            refine_metrics,
                            direction=direction,
                        )
                        candidates.append(("refine", R, t, refine_metrics, direction))
                    else:
                        _gpu_rigid_log(
                            scan_name,
                            "refine did not improve init metrics; keeping prior best candidate",
                        )
            except Exception as exc:
                import torch as _torch

                if isinstance(exc, _torch.cuda.OutOfMemoryError):
                    _gpu_rigid_log(scan_name, f"refine OOM at scales={refine_scales}: {exc}")
                    _torch.cuda.empty_cache()
                else:
                    raise

        stage, R, t, metrics, direction = max(candidates, key=lambda c: _metrics_rank(c[3]))
        _record_debug(
            debug_writer,
            debug_dir,
            scan_name,
            fixed_image,
            moving_image,
            R,
            t,
            "final",
            metrics,
            notes=f"selected {stage}",
            direction=direction,
        )
        if debug_writer is not None:
            debug_writer.set_outcome(
                True,
                message=f"accepted {stage}",
                best={"stage": stage, "metrics": metrics, "direction": direction},
            )
        _gpu_rigid_log(scan_name, f"selected pose: stage={stage}")
        return {
            "mat_path": ra._write_ants_rigid_mat(R, t),
            "R": R,
            "t": t,
            "ncc": metrics["ncc"],
            "iou": metrics["iou"],
            "initializer_record": {
                "path": opts.get("initializer", "moments"),
                "stage": stage,
                "direction": direction,
            },
            "initializer_used": opts.get("initializer", "moments"),
            "success": True,
            "debug_dir": debug_dir,
        }
    except Exception as exc:
        if debug_writer is not None:
            debug_writer.set_outcome(False, message=str(exc))
        raise GpuRigidError(str(exc)) from exc
    finally:
        if debug_writer is not None:
            debug_writer.finalize()
            _gpu_rigid_log(scan_name, f"debug artefacts → {debug_dir}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _gpu_rigid_log(scan_name, message):
    prefix = f"[GPU Rigid {scan_name}]" if scan_name else "[GPU Rigid]"
    print(f"{prefix} {message}", flush=True)
