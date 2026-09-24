"""Live GPU snapshot for the fine-tuning dialog (nvidia-smi, then torch)."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

_CACHE: Optional[Dict[str, Any]] = None
_CACHE_AT = 0.0
_CACHE_TTL_SEC = 0.75


def gpu_snapshot(*, force: bool = False) -> Dict[str, Any]:
    """Return a JSON-serializable GPU snapshot, cached for a short interval."""
    global _CACHE, _CACHE_AT
    now = time.monotonic()
    if not force and _CACHE is not None and (now - _CACHE_AT) < _CACHE_TTL_SEC:
        return _CACHE
    snapshot = _query_nvidia_smi() or _query_torch() or _cpu_fallback()
    _CACHE = snapshot
    _CACHE_AT = now
    return snapshot


def format_gib(bytes_value: Optional[int]) -> Optional[float]:
    if bytes_value is None:
        return None
    return round(float(bytes_value) / (1024.0 ** 3), 2)


def _cpu_fallback() -> Dict[str, Any]:
    return {
        "available": False,
        "kind": "cpu",
        "name": "CPU",
        "count": 0,
        "gpus": [],
        "selected": None,
        "message": "No NVIDIA GPU detected",
    }


def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not (result.stdout or "").strip():
        return None
    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            index = int(parts[0])
            total_mib = float(parts[2])
            used_mib = float(parts[3])
            free_mib = float(parts[4])
            util = None if parts[5] in ("", "[N/A]", "N/A") else float(parts[5])
            temp = None if parts[6] in ("", "[N/A]", "N/A") else float(parts[6])
        except (TypeError, ValueError):
            continue
        total_bytes = int(total_mib * 1024 * 1024)
        used_bytes = int(used_mib * 1024 * 1024)
        free_bytes = int(free_mib * 1024 * 1024)
        gpus.append({
            "index": index,
            "name": parts[1],
            "memory_total_bytes": total_bytes,
            "memory_used_bytes": used_bytes,
            "memory_free_bytes": free_bytes,
            "memory_total_gib": format_gib(total_bytes),
            "memory_used_gib": format_gib(used_bytes),
            "memory_free_gib": format_gib(free_bytes),
            "utilization_pct": util,
            "temperature_c": temp,
        })
    if not gpus:
        return None
    selected = gpus[0]
    return {
        "available": True,
        "kind": "cuda",
        "name": selected["name"],
        "count": len(gpus),
        "gpus": gpus,
        "selected": selected,
        "source": "nvidia-smi",
        "message": None,
    }


def _query_torch() -> Optional[Dict[str, Any]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        index = int(torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used_bytes = int(total_bytes) - int(free_bytes)
        allocated = int(torch.cuda.memory_allocated(index))
        selected = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "memory_total_bytes": int(total_bytes),
            "memory_used_bytes": used_bytes,
            "memory_free_bytes": int(free_bytes),
            "memory_total_gib": format_gib(int(total_bytes)),
            "memory_used_gib": format_gib(used_bytes),
            "memory_free_gib": format_gib(int(free_bytes)),
            "process_allocated_bytes": allocated,
            "process_allocated_gib": format_gib(allocated),
            "utilization_pct": None,
            "temperature_c": None,
        }
        return {
            "available": True,
            "kind": "cuda",
            "name": selected["name"],
            "count": int(torch.cuda.device_count()),
            "gpus": [selected],
            "selected": selected,
            "source": "torch",
            "message": None,
        }
    except Exception:
        return None
