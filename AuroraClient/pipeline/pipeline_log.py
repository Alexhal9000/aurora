"""Scientific Report provenance helpers (forensics + subject metadata).

Subject JSON is the per-scan source of truth for Methods reconstruction.
This module provides:
- version readers used at report-generation time
- filesystem alignment forensics (``debug_alignment``)
- ``mask_workflow`` metadata merges written when mask ops run
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def read_aurora_version() -> str:
    """Best-effort installed Aurora version (same locations as GetVersionView)."""
    version_paths = []
    system = sys.platform
    if system == "linux":
        version_paths = ["/opt/aurora-tools/version.txt"]
    elif system == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            version_paths.append(os.path.join(local_app_data, "Aurora", "version.txt"))
        version_paths.append(
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "Aurora", "version.txt")
        )
    elif system == "darwin":
        version_paths = ["/Library/Application Support/Aurora/version.txt"]

    for version_path in version_paths:
        if version_path and os.path.isfile(version_path):
            try:
                with open(version_path, "r", encoding="utf-8") as handle:
                    version = handle.read().strip()
                if version:
                    return version
            except OSError:
                continue
    return "unknown"


def read_ants_version() -> str:
    try:
        import ants  # type: ignore

        version = getattr(ants, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    return ""


_SKIP_ALIGNMENT_FORENSIC_DIRS = {
    "extracted",
    "atlas",
    "__pycache__",
    ".git",
}


def collect_alignment_forensics(directory: str) -> dict:
    """Per-subject rigid-alignment forensics for Scientific Report Methods.

    - ``likely_alpaca``: ``debug_alignment`` frames with mtime near an aligned edit.
    - ``likely_ants``: persisted ``{subject}_ants_rigid_affine.mat`` (GenericAffine rigid).
    - ``likely_dino``: leftover ``dinoreg-rigid-debug/`` folder (debug writes may be disabled).
    """
    result = {}
    if not directory or not isinstance(directory, str) or not os.path.isdir(directory):
        return result

    extracted = os.path.join(directory, "extracted")
    scan_root = extracted if os.path.isdir(extracted) else directory

    try:
        entries = os.listdir(scan_root)
    except OSError:
        return result

    for name in entries:
        if name in _SKIP_ALIGNMENT_FORENSIC_DIRS or name.startswith("."):
            continue
        subj_dir = os.path.join(scan_root, name)
        if not os.path.isdir(subj_dir):
            continue
        debug_dir = os.path.join(subj_dir, "debug_alignment")
        if not os.path.isdir(debug_dir):
            continue
        try:
            debug_files = [
                f
                for f in os.listdir(debug_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        except OSError:
            continue
        if not debug_files:
            continue

        debug_mtimes = []
        for fname in debug_files:
            try:
                debug_mtimes.append(os.path.getmtime(os.path.join(debug_dir, fname)))
            except OSError:
                continue
        if not debug_mtimes:
            continue
        debug_mtime = max(debug_mtimes)

        aligned_mtimes = []
        try:
            for fname in os.listdir(subj_dir):
                lower = fname.lower()
                if "aligned" not in lower:
                    continue
                if not (
                    ("_edit_" in lower or "_lossy_edit_" in lower)
                    and (lower.endswith(".nii.gz") or lower.endswith(".ply"))
                ):
                    continue
                try:
                    aligned_mtimes.append(os.path.getmtime(os.path.join(subj_dir, fname)))
                except OSError:
                    continue
        except OSError:
            aligned_mtimes = []

        near_aligned = False
        if aligned_mtimes:
            near_aligned = any(abs(am - debug_mtime) <= 3600.0 for am in aligned_mtimes)
        else:
            # Debug frames without a surviving aligned edit still suggest ALPACA ran.
            near_aligned = True

        result[name] = {
            "has_debug_alignment": True,
            "image_count": len(debug_files),
            "mtime_near_aligned_edit": bool(near_aligned),
            "likely_alpaca": bool(near_aligned),
        }

    # ANTs rigid: persisted GenericAffine beside subject metadata.
    for name in entries:
        if name in _SKIP_ALIGNMENT_FORENSIC_DIRS or name.startswith("."):
            continue
        if name in result and result[name].get("likely_alpaca"):
            continue
        subj_dir = os.path.join(scan_root, name)
        if not os.path.isdir(subj_dir):
            continue
        ants_affine = os.path.join(subj_dir, f"{name}_ants_rigid_affine.mat")
        if not os.path.isfile(ants_affine):
            continue
        entry = result.setdefault(name, {})
        entry["has_ants_rigid_affine"] = True
        entry["likely_ants"] = True

    # DINO-Reg: optional leftover forensic folder (writes are normally commented out).
    for name in entries:
        if name in _SKIP_ALIGNMENT_FORENSIC_DIRS or name.startswith("."):
            continue
        subj_dir = os.path.join(scan_root, name)
        if not os.path.isdir(subj_dir):
            continue
        dino_dir = os.path.join(subj_dir, "dinoreg-rigid-debug")
        if not os.path.isdir(dino_dir):
            continue
        entry = result.setdefault(name, {})
        entry["has_dinoreg_rigid_debug"] = True
        entry["likely_dino"] = True
    return result


def _subject_metadata_path(directory: str, subject: str) -> str:
    if subject == "atlas":
        # Prefer sibling atlas/ when present; fall back to extracted/atlas
        sibling = os.path.join(directory, "atlas", "atlas.json")
        if os.path.isfile(sibling):
            return sibling
        return os.path.join(directory, "extracted", "atlas", "atlas.json")
    return os.path.join(directory, "extracted", subject, f"{subject}.json")


def merge_mask_workflow_metadata(directory: str, subject: str, patch: dict) -> None:
    """Merge ``mask_workflow`` provenance into a subject's metadata JSON (best-effort).

    Expected shape (fields appear as operations run)::

        {
          "mask_workflow": {
            "creation": {
              "on": "<subject|atlas>",
              "edit": "<edit basename>",
              "method": "manual_paint",
              "ts": "..."
            },
            "propagated": true,
            "propagation": {
              "transfer_type": "reference-to-subjects",
              "source": "<reference|atlas>",
              "selected_label": 1 | null,
              "do_ai": false,
              "do_threshold": false,
              "ts": "..."
            },
            "application": {
              "operation": "masked_out" | "isolated",
              "label": 1,
              "labels": [1],
              "inserted_before_elastic": true,
              "dilation": 0,
              "ts": "..."
            }
          }
        }
    """
    if not directory or not subject or not isinstance(patch, dict) or not patch:
        return
    try:
        path = _subject_metadata_path(directory, subject)
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            return
        workflow = meta.get("mask_workflow")
        if not isinstance(workflow, dict):
            workflow = {}
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(workflow.get(key), dict):
                merged = dict(workflow[key])
                merged.update(value)
                workflow[key] = merged
            else:
                workflow[key] = value
        workflow["ts"] = datetime.now(timezone.utc).isoformat()
        meta["mask_workflow"] = workflow
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=4)
        os.replace(tmp_path, path)
    except Exception as exc:
        print(f"[mask_workflow] failed to update metadata for {subject}: {exc}")
