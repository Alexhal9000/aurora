"""Compute device choice and provenance for the GPU-first rigid tools.

Device order:
  1. CUDA (Windows / Ubuntu with an NVIDIA GPU; the installers pick the CUDA
     build the driver and GPU can run, deployment/shared/requirements_common.py).
  2. Apple GPU (MPS) on macOS, for tools that pass ``allow_mps`` (DINO-Reg).
     Ops without an MPS kernel run on the CPU for that op
     (PYTORCH_ENABLE_MPS_FALLBACK, set in AuroraClient/settings.py).
  3. The CPU, only after the user agreed to it (``allow_cpu``).

When none is usable the tool refuses to start. DINO-Reg (offer_cpu) answers
with CPU_FALLBACK_CODE; the UI then asks the user whether to run on the CPU
instead (much slower) and repeats the request with ``allow_cpu: true``.
Nothing falls back to the CPU silently. GPU Rigid has no CPU or MPS mode: off
CUDA it offers nothing over the ANTs rigid method.

The device and numeric precision actually used are recorded in the subject's
alignment provenance (``compute``), because a CPU or MPS float32 run is not
bit-identical to a CUDA float16 run of the same method.
"""

from __future__ import annotations

import sys

# Error code in the JSON answer of /align-to-reference/ when the method could
# run on the CPU if the user agrees (read by mainAurora.alignToReference).
CPU_FALLBACK_CODE = "cuda_unavailable_cpu_possible"

# Shown when an NVIDIA GPU cannot be used (same advice as the installers).
NVIDIA_DRIVER_HELP = (
    "Update the NVIDIA driver (Windows: the NVIDIA App, "
    "https://www.nvidia.com/en-us/software/nvidia-app/ ; Ubuntu: "
    "sudo ubuntu-drivers install), then reinstall Aurora."
)


# --- CUDA caching-allocator configuration -----------------------------------
# ``expandable_segments`` lets PyTorch grow one virtual segment instead of
# carving many fixed blocks, which reduces fragmentation on Linux. On Windows
# (PyTorch 2.11, WDDM driver model) it is not reliable: after a CUDA OOM the
# allocator's release path can hit
#   "!handles_.at(i) INTERNAL ASSERT FAILED at ... CUDACachingAllocator.cpp"
# which poisons the CUDA context, so the OOM can no longer be recovered by a
# smaller batch (seen 2026-11-10 in DINO-Reg on a 12 GB Windows GPU). macOS
# has no CUDA. So the option is only ever enabled on Linux.
EXPANDABLE_SEGMENTS = "expandable_segments:True"


def expandable_segments_supported() -> bool:
    """True where Aurora enables the CUDA ``expandable_segments`` allocator option."""
    return sys.platform.startswith("linux")


def configure_cuda_allocator(env=None) -> None:
    """Enable ``expandable_segments`` on Linux only; never override a user value.

    ``env`` defaults to ``os.environ`` (call before the first CUDA allocation in
    this process); pass a subprocess environment dict to configure a child.
    Off Linux it also removes the option if Aurora itself set it earlier in
    this process (the exact value below), so an older code path cannot leave
    it on.
    """
    import os

    target = os.environ if env is None else env
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if expandable_segments_supported():
            target.setdefault(key, EXPANDABLE_SEGMENTS)
        elif target.get(key) == EXPANDABLE_SEGMENTS:
            del target[key]


def is_cuda_memory_error(exc) -> bool:
    """True for any GPU memory exhaustion PyTorch reports, not only "out of memory".

    Besides ``torch.OutOfMemoryError`` ("CUDA out of memory. Tried to
    allocate ..."), a full GPU also surfaces as library handle/workspace
    failures (cuBLAS/cuDNN/cuFFT ``ALLOC_FAILED``) when those libraries
    allocate outside the caching allocator. All of them are recoverable the
    same way: free the cache and retry with less work in flight.
    """
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "alloc_failed" in msg
        or "not_enough_workspace" in msg
        or "cudaerrormemoryallocation" in msg
    )


def cuda_free_and_total_bytes(device):
    """(free, total) bytes on a CUDA device per the driver, or None elsewhere.

    ``torch.cuda.mem_get_info`` counts memory held by every process (desktop,
    browser WebGL, other tools), unlike ``memory_allocated`` which only counts
    this process's tensors.
    """
    try:
        import torch

        if torch.device(device).type != "cuda":
            return None
        free, total = torch.cuda.mem_get_info(torch.device(device))
        return int(free), int(total)
    except Exception:
        return None


def mps_available() -> bool:
    """True on a Mac whose PyTorch can use the Apple GPU."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def select_torch_device(
    tool_label: str,
    error_cls,
    *,
    allow_cpu: bool = False,
    offer_cpu: bool = True,
    allow_mps: bool = False,
):
    """Return the ``torch.device`` to run ``tool_label`` on (see module docstring).

    Raises ``error_cls`` with a message that says why no GPU is usable,
    carrying ``.code = CPU_FALLBACK_CODE`` when ``offer_cpu``.
    """
    try:
        import torch
    except ImportError as exc:
        raise error_cls(
            f"{tool_label} requires PyTorch. PyTorch is not installed."
        ) from exc

    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if allow_mps and mps_available():
        return torch.device("mps")
    if allow_cpu and offer_cpu:
        return torch.device("cpu")

    if sys.platform == "darwin":
        message = (
            f"{tool_label} runs on an NVIDIA GPU (Windows / Ubuntu), which Macs do not have."
            if not allow_mps
            else f"{tool_label} runs on the Apple GPU (Apple Silicon Mac), which this Mac "
                 "does not have or PyTorch cannot use."
        )
    elif torch.version.cuda is None:
        # CPU-only PyTorch build (older Windows installers always used it, and
        # the installers still fall back to it when no CUDA build runs here).
        message = (
            f"{tool_label} runs on an NVIDIA GPU, but this Aurora has the CPU-only "
            f"PyTorch build ({torch.__version__}). If this computer has an NVIDIA GPU: "
            f"{NVIDIA_DRIVER_HELP}"
        )
    else:
        message = (
            f"{tool_label} runs on an NVIDIA GPU, but no CUDA device is available "
            f"(PyTorch {torch.__version__}, CUDA {torch.version.cuda}). {NVIDIA_DRIVER_HELP}"
        )
    err = error_cls(message)
    if offer_cpu:
        err.code = CPU_FALLBACK_CODE
    raise err


def empty_device_cache(device) -> None:
    """Release cached GPU memory on CUDA or MPS (no-op on the CPU)."""
    try:
        import torch

        kind = torch.device(device).type
        if kind == "cuda":
            torch.cuda.empty_cache()
        elif kind == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def error_response_body(exc) -> dict:
    """JSON body for a tool error; carries ``code`` when the CPU is an option."""
    body = {"error": str(exc)}
    code = getattr(exc, "code", None)
    if code:
        body["code"] = code
    return body


def compute_provenance(device, precision: str) -> dict:
    """Serializable record of where a tool ran (alignment provenance ``compute``).

    precision: what the tool used on this device, e.g. "float16 autocast".
    """
    record = {"device": "cpu", "precision": str(precision)}
    try:
        import torch

        record["torch"] = str(torch.__version__)
        record["torch_cuda"] = torch.version.cuda
        dev = torch.device(device)
        record["device"] = dev.type
        if dev.type == "cuda" and torch.cuda.is_available():
            index = dev.index if dev.index is not None else 0
            record["gpu"] = torch.cuda.get_device_name(index)
        elif dev.type == "mps":
            import platform

            record["gpu"] = f"Apple GPU (MPS), {platform.machine()}"
    except Exception:
        pass
    return record
