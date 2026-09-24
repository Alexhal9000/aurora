"""Preflight validation of a fine-tuning experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..aiModels.contracts import (
    FOUNDATION_MEDSAM2,
    PARADIGM_DIRECT,
    PARADIGM_PROMPT_GUIDED,
    PRESET_DECODER_FOCUSED,
    PRESET_ENCODER_DECODER,
    PRESET_FULL,
    DIMENSIONALITY_2D,
    DIMENSIONALITY_3D,
)
from ..aiModels.paths import ensure_models_root
from ..aiModels.registry import get_model
from .discover import discover_project
from .gpu_monitor import gpu_snapshot
from .splits import (
    SplitError,
    assert_no_subject_leakage,
    assignments_from_user,
    k_fold_splits,
    subject_level_split,
)


class ValidationReport:
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.invalid_subjects: List[Dict[str, Any]] = []
        self.assignments = None
        self.device = None

    @property
    def ok(self) -> bool:
        return not self.errors and not self.invalid_subjects

    def add_error(self, code: str, message: str, **extra):
        item = {"code": code, "message": message}
        item.update(extra)
        self.errors.append(item)

    def add_warning(self, code: str, message: str, **extra):
        item = {"code": code, "message": message}
        item.update(extra)
        self.warnings.append(item)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "invalid_subjects": self.invalid_subjects,
            "assignments": self.assignments,
            "device": self.device,
        }


def validate_experiment(payload: Dict[str, Any], *, discovered: Optional[Dict[str, Any]] = None) -> ValidationReport:
    report = ValidationReport()
    directory = payload.get("directory")
    if not directory:
        report.add_error("missing_directory", "Project directory is required")
        return report

    try:
        discovered = discovered or discover_project(directory)
    except Exception as exc:
        report.add_error("discover_failed", str(exc))
        return report

    subject_map = {item["name"]: item for item in discovered.get("subjects") or []}
    selected_subjects = _unique(payload.get("subjects") or [])
    selected_labels = _int_list(payload.get("label_ids") or payload.get("labels") or [])
    if not selected_subjects:
        report.add_error("no_subjects", "Select at least one subject")
    if not selected_labels:
        report.add_error("no_labels", "Select at least one label")

    for name in selected_subjects:
        record = subject_map.get(name)
        if record is None:
            report.invalid_subjects.append({
                "name": name,
                "issues": ["subject_not_in_project"],
            })
            continue
        issues = list(record.get("issues") or [])
        if not record.get("image_filename"):
            issues.append("image_missing")
        if not record.get("mask_filename"):
            issues.append("mask_missing")
        known_labels = record.get("label_ids") or []
        missing_labels = []
        if known_labels:
            missing_labels = [label for label in selected_labels if label not in known_labels]
        if missing_labels:
            issues.append("selected_labels_absent")
        blocking = [issue for issue in issues if issue not in {"faulty"}]
        if missing_labels or blocking:
            report.invalid_subjects.append({
                "name": name,
                "issues": sorted(set(issues)),
                "missing_labels": missing_labels,
            })

    if report.invalid_subjects:
        report.add_error(
            "invalid_subjects",
            "Some selected subjects cannot be used. Remove them before training.",
            subjects=[item["name"] for item in report.invalid_subjects],
        )

    model_id = payload.get("foundation_model") or payload.get("model") or FOUNDATION_MEDSAM2
    descriptor = get_model(model_id)
    if descriptor is None:
        report.add_error("unknown_model", f"Model {model_id} is not available")
        return report

    caps = descriptor.get("capabilities") or {}
    dimensionality = payload.get("dimensionality") or descriptor.get("dimensionality")
    paradigm = payload.get("paradigm") or descriptor.get("paradigm")
    supported_dims = set(caps.get("supported_dimensionalities") or [descriptor.get("dimensionality")])
    supported_paradigms = set(caps.get("supported_paradigms") or [descriptor.get("paradigm")])
    if dimensionality not in supported_dims:
        report.add_error(
            "unsupported_dimensionality",
            f"{descriptor.get('display_name')} does not support dimensionality {dimensionality}",
        )
    if paradigm not in supported_paradigms:
        reason = caps.get("direct_unavailable_reason") if paradigm == PARADIGM_DIRECT else (
            caps.get("two_d_unavailable_reason") if dimensionality == DIMENSIONALITY_2D else
            f"{descriptor.get('display_name')} does not support paradigm {paradigm}"
        )
        report.add_error("unsupported_paradigm", reason or f"Unsupported paradigm {paradigm}")

    preset = payload.get("trainable_preset") or PRESET_DECODER_FOCUSED
    legal_presets = {item["id"] for item in (caps.get("trainable_presets") or [])} or {
        PRESET_DECODER_FOCUSED, PRESET_ENCODER_DECODER, PRESET_FULL
    }
    if preset not in legal_presets:
        report.add_error("unsupported_preset", f"Trainable preset {preset} is not supported")

    strategy = payload.get("evaluation_strategy") or "train_val_test"
    seed = int(payload.get("seed") or caps.get("default_seed") or 123)
    n = len(selected_subjects) - len({item["name"] for item in report.invalid_subjects})
    valid_names = [
        name for name in selected_subjects
        if name not in {item["name"] for item in report.invalid_subjects}
    ]

    try:
        if strategy == "user":
            report.assignments = assignments_from_user(
                payload.get("train_subjects") or [],
                payload.get("validation_subjects") or [],
                payload.get("test_subjects") or [],
            )
        elif strategy == "kfold":
            k = int(payload.get("k_folds") or 5)
            if not caps.get("supports_cross_validation"):
                report.add_error("unsupported_strategy", "This model does not support cross-validation")
            elif n < k:
                report.add_error(
                    "insufficient_subjects_cv",
                    f"Need at least {k} subjects for {k}-fold cross-validation, got {n}",
                )
            else:
                holdout_pct = float(payload.get("test_pct") or 0)
                holdout = []
                pool = valid_names
                if holdout_pct > 0 and n >= k + 1:
                    holdout_n = max(1, int(round(n * holdout_pct / 100.0)))
                    holdout = valid_names[:holdout_n]
                    pool = valid_names[holdout_n:]
                folds = k_fold_splits(pool, k=k, seed=seed, holdout_test=holdout)
                report.assignments = {"strategy": "kfold", "k": k, "folds": folds}
        else:
            defaults = caps.get("default_split") or {"train": 70, "validation": 15, "test": 15}
            train_pct = float(payload.get("train_pct", defaults["train"]))
            val_pct = float(payload.get("val_pct", defaults["validation"]))
            test_pct = float(payload.get("test_pct", defaults["test"]))
            allow_no_test = bool(payload.get("allow_no_test"))
            min_needed = 3 if test_pct > 0 and not allow_no_test else 2
            if n < min_needed:
                report.add_error(
                    "insufficient_subjects",
                    f"Need at least {min_needed} subjects for this evaluation strategy, got {n}",
                )
            else:
                report.assignments = subject_level_split(
                    valid_names,
                    train_pct=train_pct,
                    val_pct=val_pct,
                    test_pct=test_pct,
                    seed=seed,
                    allow_no_test=allow_no_test or test_pct <= 0,
                )
                report.assignments["strategy"] = "train_val_test"
        if report.assignments and "folds" not in (report.assignments or {}):
            assert_no_subject_leakage({
                key: report.assignments.get(key) or []
                for key in ("train", "validation", "test")
            })
    except SplitError as exc:
        report.add_error("invalid_split", str(exc))

    report.device = gpu_snapshot()
    if report.device.get("kind") == "cpu" or not report.device.get("available"):
        report.add_warning(
            "cpu_training",
            "No CUDA GPU was detected. Fine-tuning on CPU is supported but likely impractical.",
        )

    try:
        root = ensure_models_root()
        if not os.access(root, os.W_OK):
            report.add_error("output_not_writable", f"Cannot write to {root}")
    except OSError as exc:
        report.add_error("output_unavailable", str(exc))

    display_name = (payload.get("display_name") or "").strip()
    if not display_name:
        report.add_warning("missing_display_name", "A default model name will be generated")

    return report


def _device_info() -> Dict[str, Any]:
    return gpu_snapshot()


def _unique(values: Sequence[Any]) -> List[str]:
    seen = []
    used = set()
    for value in values:
        text = str(value).strip()
        if not text or text in used:
            continue
        used.add(text)
        seen.append(text)
    return seen


def _int_list(values: Sequence[Any]) -> List[int]:
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("id")
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 255 and number not in result:
            result.append(number)
    return result
