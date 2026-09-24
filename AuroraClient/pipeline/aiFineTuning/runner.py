"""Launch, monitor, cancel, and register fine-tuning runs."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ..aiModels.contracts import FOUNDATION_MEDSAM2, medsam2_train_capabilities
from ..aiModels.foundation_models import MEDSAM2_ID, require_installed
from ..aiModels.medsam2_adapter import DEFAULT_CONFIG, default_checkpoint
from ..aiModels.paths import ensure_models_root, registered_model_dir, run_dir
from .checkpoint import INFERENCE_ARTIFACT_FILENAME, export_inference_checkpoint
from .dataset import collect_examples
from .discover import discover_project
from .metadata import build_model_metadata, default_display_name
from .storage import (
    EVAL_FILENAME,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    TERMINAL_STATUSES,
    atomic_write_json,
    checkpoints_dir,
    logs_dir,
    read_json,
    read_pid,
    read_run_config,
    read_run_status,
    request_cancel,
    write_pid,
    write_run_config,
    write_run_status,
)
from .validate import validate_experiment

_PROCESSES = {}
_WATCHERS = {}


def start_run(payload: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
    require_installed(MEDSAM2_ID)
    discovered = discover_project(payload["directory"])
    report = validate_experiment(payload, discovered=discovered)
    if not report.ok:
        return {"ok": False, "validation": report.as_dict()}

    run_id = str(uuid.uuid4())
    ensure_models_root()
    logs_dir(run_id)
    checkpoints_dir(run_id)

    subject_records = {item["name"]: item for item in discovered["subjects"]}
    labels = _selected_labels(payload, discovered)
    assignments = report.assignments or {}
    label_ids = [item["id"] for item in labels]
    fold_specs = _fold_specs(
        payload["directory"],
        assignments,
        label_ids,
        subject_records,
    )
    first_fold = fold_specs[0] if fold_specs else {
        "examples_train": [],
        "examples_validation": [],
        "examples_test": [],
    }
    examples_train = first_fold["examples_train"]
    examples_val = first_fold["examples_validation"]
    examples_test = first_fold["examples_test"]

    caps = medsam2_train_capabilities()
    hp_defaults = {key: spec.get("default") for key, spec in caps["hyperparameters"].items()}
    hp = dict(hp_defaults)
    hp.update(payload.get("hyperparameters") or {})

    display_name = (payload.get("display_name") or "").strip() or default_display_name(
        discovered.get("project_name") or "",
        [item["name"] for item in labels],
    )

    config = {
        "run_id": run_id,
        "directory": discovered["directory"],
        "project_name": discovered.get("project_name"),
        "foundation_model": payload.get("foundation_model") or FOUNDATION_MEDSAM2,
        "dimensionality": payload.get("dimensionality") or "3D",
        "paradigm": payload.get("paradigm") or "prompt_guided",
        "display_name": display_name,
        "subjects": payload.get("subjects"),
        "labels": labels,
        "assignments": assignments,
        "examples": {
            "train": [ _example_ref(item) for item in examples_train ],
            "validation": [ _example_ref(item) for item in examples_val ],
            "test": [ _example_ref(item) for item in examples_test ],
        },
        "examples_train": examples_train,
        "examples_validation": examples_val,
        "examples_test": examples_test,
        "folds": fold_specs,
        "subject_provenance": {
            name: {
                "image_filename": subject_records[name].get("image_filename"),
                "mask_filename": subject_records[name].get("mask_filename"),
                "voxel_size": subject_records[name].get("voxel_size"),
                "affine_zooms": subject_records[name].get("affine_zooms"),
                "alignment_to": subject_records[name].get("alignment_to"),
                "elastic_to": subject_records[name].get("elastic_to"),
                "intensity_value_mapping": subject_records[name].get("intensity_value_mapping"),
            }
            for name in (payload.get("subjects") or [])
            if name in subject_records
        },
        "prompt": (payload.get("prompt") or {
            "prompt_type": "mask_and_box",
            "initialization": "middle_slab_random",
        }),
        "trainable_preset": payload.get("trainable_preset") or "decoder_focused",
        "hyperparameters": hp,
        "augmentation": payload.get("augmentation") or {"enabled": False},
        "seed": int(payload.get("seed") or hp.get("seed") or 123),
        "loss_weights": caps.get("loss_weights"),
        "base_checkpoint": payload.get("base_checkpoint") or default_checkpoint(),
        "config_path": payload.get("config_path") or DEFAULT_CONFIG,
        "device": report.device,
        "dry_run": bool(dry_run or payload.get("dry_run")),
    }
    write_run_config(run_id, config)
    write_run_status(run_id, {
        "status": STATUS_RUNNING,
        "phase": "setup",
        "epoch": 0,
        "epochs": hp.get("epochs"),
        "progress": 0.0,
        "message": "Starting training",
    })
    _spawn_worker(run_id)
    _start_watcher(run_id)
    return {"ok": True, "run_id": run_id, "validation": report.as_dict(), "config_summary": {
        "display_name": display_name,
        "n_train": len(examples_train),
        "n_validation": len(examples_val),
        "n_test": len(examples_test),
        "assignments": assignments,
    }}


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    status = read_run_status(run_id)
    if status is None:
        return None
    config = read_run_config(run_id) or {}
    from .gpu_monitor import gpu_snapshot
    from .preview_store import read_manifest, run_preview_dir
    preview = read_manifest(run_preview_dir(run_id))
    return {
        "run_id": run_id,
        "status": status,
        "display_name": config.get("display_name"),
        "foundation_model": config.get("foundation_model"),
        "can_register": status.get("status") == STATUS_SUCCEEDED and bool(status.get("best_checkpoint")),
        "gpu": gpu_snapshot(),
        "preview": preview,
    }


def cancel_run(run_id: str) -> Dict[str, Any]:
    request_cancel(run_id)
    proc = _PROCESSES.get(run_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    status = read_run_status(run_id) or {}
    if status.get("status") == STATUS_RUNNING:
        status = {
            **status,
            "status": STATUS_CANCELLED,
            "message": "Cancelled by user",
        }
        write_run_status(run_id, status)
    return {"ok": True, "run_id": run_id, "status": status}


def register_run(run_id: str, display_name: Optional[str] = None) -> Dict[str, Any]:
    status = read_run_status(run_id)
    config = read_run_config(run_id)
    if not status or not config:
        raise FileNotFoundError(f"Training run {run_id} was not found")
    if status.get("status") != STATUS_SUCCEEDED:
        raise ValueError("Only a successful training run can be registered")
    checkpoint = status.get("best_checkpoint") or (status.get("selected_checkpoint") or {}).get("path")
    if not checkpoint or not os.path.isfile(checkpoint):
        raise ValueError("Selected checkpoint is missing; the run will not be registered")

    model_id = str(uuid.uuid4())
    dest = registered_model_dir(model_id)
    dest.mkdir(parents=True, exist_ok=True)
    artifact = dest / INFERENCE_ARTIFACT_FILENAME
    export_inference_checkpoint(checkpoint, artifact)

    evaluation = {
        "validation": status.get("val_metrics") or (status.get("selected_checkpoint") or {}).get("validation"),
        "test": status.get("test_metrics") or (status.get("selected_checkpoint") or {}).get("test"),
        "history": status.get("history"),
        "fold_results": status.get("fold_results"),
        "selected_fold": status.get("selected_fold"),
        "foundation_vs_finetuned": status.get("foundation_vs_finetuned"),
        "figures": status.get("figures"),
    }
    metadata = build_model_metadata(
        model_id=model_id,
        display_name=(display_name or config.get("display_name") or model_id).strip(),
        run_id=run_id,
        config=config,
        evaluation=evaluation,
        selected_checkpoint=status.get("selected_checkpoint") or {
            "path": str(artifact),
            "epoch": status.get("best_epoch"),
            "criterion": "val_loss_gated_dice",
            "validation": {
                "dice": status.get("best_val_dice"),
                "loss": status.get("best_val_loss"),
            },
        },
    )
    atomic_write_json(dest / "metadata.json", metadata)
    atomic_write_json(run_dir(run_id) / EVAL_FILENAME, evaluation)
    src_figures = run_dir(run_id) / "figures"
    if src_figures.is_dir():
        shutil.copytree(src_figures, dest / "figures", dirs_exist_ok=True)
    src_timing = run_dir(run_id) / "logs"
    if src_timing.is_dir():
        dest_logs = dest / "logs"
        dest_logs.mkdir(parents=True, exist_ok=True)
        for name in ("timing.txt", "timing.json", "timing.jsonl"):
            src = src_timing / name
            if src.is_file():
                shutil.copy2(src, dest_logs / name)
    src_preview = run_dir(run_id) / "preview"
    if src_preview.is_dir():
        shutil.copytree(src_preview, dest / "preview", dirs_exist_ok=True)
    return {
        "ok": True,
        "model_id": model_id,
        "display_name": metadata["identity"]["display_name"],
        "path": str(dest),
        "metadata": metadata,
    }


def _spawn_worker(run_id: str) -> None:
    import subprocess

    client_root = Path(__file__).resolve().parents[2]  # AuroraClient/
    cmd = [
        sys.executable,
        "-m",
        "pipeline.aiFineTuning.train_worker",
        "--run-dir",
        str(run_dir(run_id)),
    ]
    log_path = logs_dir(run_id) / "worker.log"
    handle = open(log_path, "ab")
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(client_root), pythonpath]) if pythonpath else str(client_root)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(
        cmd,
        cwd=str(client_root),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    _PROCESSES[run_id] = proc
    write_pid(run_id, proc.pid)


def _start_watcher(run_id: str) -> None:
    if run_id in _WATCHERS:
        return
    thread = threading.Thread(target=_watch_run, args=(run_id,), daemon=True)
    _WATCHERS[run_id] = thread
    thread.start()


def _watch_run(run_id: str) -> None:
    last = None
    while True:
        status = read_run_status(run_id) or {}
        snapshot = (
            status.get("status"),
            status.get("epoch"),
            status.get("train_loss"),
            status.get("best_val_dice"),
            status.get("best_val_loss"),
            status.get("message"),
            status.get("phase"),
        )
        if snapshot != last:
            _broadcast(run_id, status)
            last = snapshot
        if status.get("status") in TERMINAL_STATUSES:
            proc = _PROCESSES.get(run_id)
            if proc is not None and proc.poll() is None:
                time.sleep(0.5)
            break
        proc = _PROCESSES.get(run_id)
        if proc is not None and proc.poll() is not None and status.get("status") == STATUS_RUNNING:
            # Worker died without a terminal status.
            write_run_status(run_id, {
                **status,
                "status": STATUS_FAILED,
                "error": f"Training process exited with code {proc.returncode}",
            })
            _broadcast(run_id, read_run_status(run_id) or {})
            break
        time.sleep(1.0)


def _broadcast(run_id: str, status: Dict[str, Any]) -> None:
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer
        layer = get_channel_layer()
        if layer is None:
            return
        epoch = status.get("epoch") or 0
        epochs = status.get("epochs") or 1
        message = status.get("message")
        if not message:
            parts = [f"Fine-tune {status.get('status', '')}"]
            if epoch:
                parts.append(f"epoch {epoch}/{epochs}")
            if status.get("train_loss") is not None:
                parts.append(f"loss {status['train_loss']:.4f}")
            if status.get("best_val_loss") is not None:
                parts.append(f"best val loss {status['best_val_loss']:.4f}")
            if status.get("best_val_dice") is not None:
                parts.append(f"gated Dice {status['best_val_dice']:.3f}")
            message = " · ".join(parts)
        async_to_sync(layer.group_send)(
            "progress_group",
            {
                "type": "send_progress",
                "progress": float(status.get("progress") or 0),
                "scan_name": run_id,
                "custom_message": message,
                "total": int(epochs),
                "current": int(epoch),
                "run_id": run_id,
                "train_loss": status.get("train_loss"),
                "val_metrics": status.get("val_metrics"),
                "best_checkpoint": status.get("best_checkpoint"),
                "run_status": status.get("status"),
            },
        )
    except Exception:
        return


def _selected_labels(payload: Dict[str, Any], discovered: Dict[str, Any]) -> list:
    wanted = payload.get("label_ids") or payload.get("labels") or []
    ids = []
    for item in wanted:
        if isinstance(item, dict):
            ids.append(int(item.get("id")))
        else:
            ids.append(int(item))
    names = discovered.get("label_names") or {}
    return [
        {"id": label_id, "name": names.get(str(label_id)) or f"Label {label_id}"}
        for label_id in ids
    ]


def _fold_specs(
    directory: str,
    assignments: Dict[str, Any],
    label_ids: list,
    subject_records: Dict[str, Any],
) -> list:
    if assignments.get("strategy") == "kfold":
        folds = assignments.get("folds") or []
    else:
        folds = [{
            "fold": 0,
            "train": assignments.get("train") or [],
            "validation": assignments.get("validation") or [],
            "test": assignments.get("test") or [],
        }]
    packed = []
    for fold in folds:
        train_examples = collect_examples(directory, fold.get("train") or [], label_ids, subject_records)
        val_examples = collect_examples(directory, fold.get("validation") or [], label_ids, subject_records)
        test_examples = collect_examples(directory, fold.get("test") or [], label_ids, subject_records)
        packed.append({
            "fold": int(fold.get("fold") or 0),
            "train_subjects": list(fold.get("train") or []),
            "validation_subjects": list(fold.get("validation") or []),
            "test_subjects": list(fold.get("test") or []),
            "examples_train": train_examples,
            "examples_validation": val_examples,
            "examples_test": test_examples,
            "examples": {
                "train": [_example_ref(item) for item in train_examples],
                "validation": [_example_ref(item) for item in val_examples],
                "test": [_example_ref(item) for item in test_examples],
            },
        })
    return packed


def _example_ref(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject": example.get("subject"),
        "label_id": example.get("label_id"),
        "image_filename": example.get("image_filename"),
        "mask_filename": example.get("mask_filename"),
    }
