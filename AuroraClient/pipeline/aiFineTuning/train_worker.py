"""Subprocess training worker. Invoked as ``python -m pipeline.aiFineTuning.train_worker``."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .checkpoint import CRITERION, CRITERION_FOLDS, fold_rank_key, should_save_checkpoint
from .validation_mode import (
    MODE_FAST,
    dice_source_for_mode,
    resolve_validation_mode,
    volume_val_examples,
)

# Must be set before torch is imported in this subprocess.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aurora MedSAM2 fine-tuning worker")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    run_id = run_dir.name

    # Make MedSAM2 importable the same way inference does.
    medsam2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MedSAM2"))
    if medsam2_dir not in sys.path:
        sys.path.insert(0, medsam2_dir)

    from .evaluate import aggregate_metrics, binary_metrics
    from .dataset import augment_clip, build_clip, load_example_arrays
    from .storage import (
        STATUS_CANCELLED,
        STATUS_FAILED,
        STATUS_RUNNING,
        STATUS_SUCCEEDED,
        cancel_requested,
        checkpoints_dir,
        read_json,
        write_run_status,
    )

    config = read_json(run_dir / "config.json")
    if not config:
        return _fail(run_id, "Missing immutable training config")

    if config.get("dry_run"):
        return _dry_run(run_id, config)

    try:
        return _train(run_id, config, medsam2_dir)
    except Exception as exc:
        write_run_status(run_id, {
            "status": STATUS_FAILED,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 1


def _fail(run_id: str, message: str) -> int:
    from .storage import STATUS_FAILED, write_run_status
    write_run_status(run_id, {"status": STATUS_FAILED, "error": message})
    return 1


def _dry_run(run_id: str, config: dict) -> int:
    from .figures import figures_dir, save_dice_curves, save_fold_val_summary, save_loss_curves
    from .storage import STATUS_SUCCEEDED, checkpoints_dir, write_run_status

    ckpt_dir = checkpoints_dir(run_id)
    fake = ckpt_dir / "epoch_001.pt"
    fake.write_bytes(b"dry-run-checkpoint")
    history = [{
        "epoch": 1,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "train_dice": 1.0,
        "val_dice": 1.0,
        "val_metrics": {"dice": 1.0, "iou": 1.0},
    }]
    fig_dir = figures_dir(run_id)
    loss_fig = save_loss_curves(fig_dir / "training_loss.png", history)
    dice_fig = save_dice_curves(fig_dir / "training_dice.png", history)
    folds = config.get("folds") or []
    fold_results = []
    for spec in folds:
        fold_results.append({
            "fold": spec.get("fold", 0),
            "best_val_dice": 1.0,
            "best_val_loss": 0.0,
            "best_epoch": 1,
            "best_checkpoint": str(fake),
            "selected": int(spec.get("fold") or 0) == 0,
        })
    extra_figures = []
    if len(fold_results) > 1:
        extra_figures.append(str(save_fold_val_summary(fig_dir / "fold_val_dice.png", fold_results)))
    write_run_status(run_id, {
        "status": STATUS_SUCCEEDED,
        "epoch": 1,
        "epochs": 1,
        "train_loss": 0.0,
        "best_checkpoint": str(fake),
        "best_epoch": 1,
        "best_val_dice": 1.0,
        "best_val_loss": 0.0,
        "dry_run": True,
        "selected_fold": 0 if fold_results else None,
        "fold_results": fold_results or None,
        "history": history,
        "figures": [str(loss_fig), str(dice_fig)] + extra_figures,
        "selected_checkpoint": {
            "path": str(fake),
            "epoch": 1,
            "criterion": CRITERION_FOLDS if len(fold_results) > 1 else CRITERION,
            "fold": 0 if fold_results else None,
            "validation": {"dice": 1.0, "loss": 0.0},
        },
    })
    return 0


def _emit(run_id: str, state: dict, **fields) -> dict:
    """Overwrite status.json with merged fields so the UI can show each section."""
    from .gpu_monitor import gpu_snapshot
    from .storage import STATUS_RUNNING, write_run_status
    state.update(fields)
    state.setdefault("status", STATUS_RUNNING)
    state["gpu"] = gpu_snapshot()
    write_run_status(run_id, dict(state))
    return state


def _train(run_id: str, config: dict, medsam2_dir: str) -> int:
    import torch
    import torch.nn.functional as F
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from ..aiModels.medsam2_adapter import apply_trainable_preset
    from .dataset import augment_clip, build_clip, load_example_arrays
    from .evaluate import aggregate_metrics, binary_metrics
    from .storage import (
        STATUS_CANCELLED,
        STATUS_RUNNING,
        STATUS_SUCCEEDED,
        cancel_requested,
        checkpoints_dir,
        write_run_status,
    )

    original_cwd = os.getcwd()
    hydra_cm = None
    timing = None
    os.chdir(medsam2_dir)
    state = {
        "status": STATUS_RUNNING,
        "phase": "setup",
        "progress": 0.0,
        "message": "Loading MedSAM2 configuration",
    }
    try:
        write_run_status(run_id, dict(state))
        if medsam2_dir not in sys.path:
            sys.path.insert(0, medsam2_dir)
        GlobalHydra.instance().clear()
        config_dir = os.path.join(medsam2_dir, "configs")
        hydra_cm = initialize_config_dir(version_base=None, config_dir=config_dir)
        hydra_cm.__enter__()
        from sam2.sam2_video_trainer import SAM2VideoTrainer

        device_kind = (config.get("device") or {}).get("kind") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device("cuda" if device_kind == "cuda" and torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        hp = config.get("hyperparameters") or {}
        seed = int(config.get("seed") or hp.get("seed") or 123)
        torch.manual_seed(seed)
        np.random.seed(seed)
        epochs = int(hp.get("epochs") or 75)
        state["epochs"] = epochs
        state["epoch"] = 0
        from .timing import TimingLog
        timing = TimingLog(run_id, extra={
            "device": str(device),
            "epochs": epochs,
            "preset": config.get("trainable_preset") or "decoder_focused",
            "prompt": (config.get("prompt") or {}).get("initialization"),
        })

        _emit(run_id, state, phase="setup", progress=0.08, message="Loading foundation checkpoint")
        checkpoint = config.get("base_checkpoint")
        config_name = os.path.basename(config.get("config_path") or "sam2.1_hiera_t512.yaml")
        trainer = SAM2VideoTrainer(config_name, checkpoint, device=device)

        preset = config.get("trainable_preset") or "decoder_focused"
        _emit(run_id, state, phase="setup", progress=0.2, message=f"Selecting trainable layers ({preset})")
        flags = apply_trainable_preset(trainer.model, preset)

        _emit(run_id, state, phase="setup", progress=0.28, message="Snapshotting foundation weights")
        foundation_cpu = {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.state_dict().items()
        }

        fold_specs = config.get("folds") or [{
            "fold": 0,
            "examples_train": config.get("examples_train") or [],
            "examples_validation": config.get("examples_validation") or [],
            "examples_test": config.get("examples_test") or [],
        }]
        n_folds = max(len(fold_specs), 1)
        kfold = n_folds > 1 or (config.get("assignments") or {}).get("strategy") == "kfold"
        prompt = config.get("prompt") or {}

        aug = config.get("augmentation") or {}
        loss_weights = config.get("loss_weights") or {"loss_mask": 20.0, "loss_dice": 1.0, "loss_iou": 1.0}
        val_every = int(hp.get("validation_frequency") or 1)
        patience = int(hp.get("early_stopping_patience") or 0)
        use_amp = bool(hp.get("amp", True)) and device.type == "cuda"
        clip_norm = float(hp.get("grad_clip_max_norm") or 0.1)
        num_frames = int(hp.get("num_frames") or 8)
        resolution = int(hp.get("resolution") or 512)
        ckpt_dir = checkpoints_dir(run_id)

        fold_results = []
        selected = None
        for fold_i, spec in enumerate(fold_specs):
            _restore_cpu(trainer.model, foundation_cpu, device)
            apply_trainable_preset(trainer.model, preset)
            optimizer, scheduler = _make_optimizer(trainer.model, hp, epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            rng = np.random.default_rng(seed + int(spec.get("fold") or fold_i))
            result = _train_fold(
                run_id=run_id,
                state=state,
                trainer=trainer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                spec=spec,
                fold_i=fold_i,
                n_folds=n_folds,
                epochs=epochs,
                device=device,
                aug=aug,
                rng=rng,
                loss_weights=loss_weights,
                val_every=val_every,
                patience=patience,
                use_amp=use_amp,
                clip_norm=clip_norm,
                num_frames=num_frames,
                resolution=resolution,
                ckpt_dir=ckpt_dir,
                flags=flags,
                prompt=prompt,
                hyperparameters=hp,
                timing=timing,
            )
            if result.get("cancelled"):
                return 0
            fold_results.append(result)
            if selected is None or fold_rank_key(result) < fold_rank_key(selected):
                selected = result

        if selected is None or not selected.get("best_checkpoint"):
            return _fail(run_id, "Training finished without a checkpoint")

        for row in fold_results:
            row["selected"] = row.get("fold") == selected.get("fold") and row.get("best_checkpoint") == selected.get("best_checkpoint")

        from .figures import (
            figures_dir,
            save_before_after_bars,
            save_dice_curves,
            save_fold_val_summary,
            save_loss_curves,
            save_validation_multiview,
        )
        from .storage import atomic_write_json
        from .dataset import load_example_arrays
        from .volume_inference import predict_binary_volume
        from .evaluate import binary_metrics, aggregate_metrics
        from .preview_store import run_preview_dir, save_epoch_prediction, subject_key, update_manifest

        fig_dir = figures_dir(run_id)
        figure_paths = []
        if kfold:
            for row in fold_results:
                fold_n = int(row.get("fold") or 0)
                history = row.get("history") or []
                figure_paths.append(str(save_loss_curves(
                    fig_dir / f"fold_{fold_n:02d}_training_loss.png",
                    history,
                    title=f"Fold {fold_n + 1} training / validation loss",
                )))
                figure_paths.append(str(save_dice_curves(
                    fig_dir / f"fold_{fold_n:02d}_training_dice.png",
                    history,
                    title=f"Fold {fold_n + 1} training / validation Dice",
                )))
            figure_paths.append(str(save_fold_val_summary(fig_dir / "fold_val_dice.png", fold_results)))
        else:
            history = selected.get("history") or []
            figure_paths.append(str(save_loss_curves(
                fig_dir / "training_loss.png",
                history,
                title="Training / validation loss",
            )))
            figure_paths.append(str(save_dice_curves(
                fig_dir / "training_dice.png",
                history,
                title="Training / validation Dice",
            )))

        val_examples = selected.get("examples_validation") or []
        comparison = None
        if val_examples:
            val_dir = fig_dir / "validation"
            val_dir.mkdir(parents=True, exist_ok=True)
            comparison = []
            overlay_paths = []
            with timing.span("compare", n_val=len(val_examples)):
              for index, example in enumerate(val_examples):
                subject = example.get("subject") or f"case_{index}"
                label_id = example.get("label_id")
                _emit(
                    run_id, state,
                    phase="compare",
                    progress=0.90 + 0.08 * (index / max(len(val_examples), 1)),
                    message=f"Validation 3D figure {index + 1}/{len(val_examples)}: {subject}",
                )
                image, binary = load_example_arrays(example)
                _restore_cpu(trainer.model, foundation_cpu, device)
                trainer.model.eval()
                pred_f = predict_binary_volume(
                    trainer, image, binary, device, resolution, prompt=prompt
                )
                _load_checkpoint(trainer.model, selected["best_checkpoint"], device)
                trainer.model.eval()
                pred_t = predict_binary_volume(
                    trainer, image, binary, device, resolution, prompt=prompt
                )
                metrics_f = binary_metrics(pred_f, binary, include_hd95=False)
                metrics_t = binary_metrics(pred_t, binary, include_hd95=False)
                comparison.append({
                    "subject": subject,
                    "label_id": label_id,
                    "foundation_dice": metrics_f.get("dice"),
                    "finetuned_dice": metrics_t.get("dice"),
                    "foundation_iou": metrics_f.get("iou"),
                    "finetuned_iou": metrics_t.get("iou"),
                })
                stem = "".join(
                    ch if ch.isalnum() or ch in "-_." else "_"
                    for ch in f"{subject}_label{label_id}"
                )
                overlay_paths.append(str(save_validation_multiview(
                    val_dir / f"{stem}_foundation_vs_finetuned.png",
                    image,
                    pred_f,
                    pred_t,
                    subject=subject,
                    label_id=label_id,
                    gt_mask=binary,
                )))
                if resolve_validation_mode(selected.get("validation_mode") or hp.get("validation_mode")) != MODE_FAST:
                    preview_info = save_epoch_prediction(
                        run_preview_dir(run_id),
                        example,
                        image,
                        pred_t,
                        epoch=int(selected.get("best_epoch") or 0),
                        dice=metrics_t.get("dice"),
                    )
                    update_manifest(
                        run_preview_dir(run_id),
                        sticky_key=subject_key(example) if index == 0 else None,
                        subject_info=preview_info,
                        epoch=int(selected.get("best_epoch") or 0),
                        preview_kind="volume_3d",
                    )
                del image, binary, pred_f, pred_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            figure_paths.append(str(save_before_after_bars(fig_dir / "validation_before_after.png", comparison)))
            figure_paths.extend(overlay_paths)
            foundation_metrics = aggregate_metrics([
                {"dice": row["foundation_dice"], "iou": row["foundation_iou"]}
                for row in comparison
            ])
            finetuned_metrics = aggregate_metrics([
                {"dice": row["finetuned_dice"], "iou": row["finetuned_iou"]}
                for row in comparison
            ])
            atomic_write_json(fig_dir / "validation_before_after.json", {
                "foundation": foundation_metrics,
                "finetuned": finetuned_metrics,
                "per_example": comparison,
            })
            selected["val_metrics"] = finetuned_metrics
            comparison_summary = {
                "foundation": foundation_metrics,
                "finetuned": finetuned_metrics,
            }
        else:
            comparison_summary = None

        test_metrics = None
        test_examples = selected.get("examples_test") or []
        if selected.get("best_checkpoint") and test_examples:
            _emit(run_id, state, phase="test", progress=0.98, message="Scoring the held-out test set")
            _load_checkpoint(trainer.model, selected["best_checkpoint"], device)
            with timing.span("test", n=len(test_examples)):
                test_metrics = _evaluate_split(
                    trainer, test_examples, device, num_frames, resolution, prompt=prompt,
                    batch_size=selected.get("batch_size") or 1,
                    use_amp=use_amp,
                )["metrics"]

        criterion = CRITERION_FOLDS if kfold else CRITERION
        timing_payload = timing.flush()
        write_run_status(run_id, {
            **state,
            "status": STATUS_SUCCEEDED,
            "phase": "done",
            "epoch": selected.get("best_epoch"),
            "epochs": epochs,
            "history": selected.get("history"),
            "fold_results": [
                {
                    "fold": row.get("fold"),
                    "best_epoch": row.get("best_epoch"),
                    "best_val_dice": row.get("best_val_dice"),
                    "best_val_loss": row.get("best_val_loss"),
                    "best_checkpoint": row.get("best_checkpoint"),
                    "selected": row.get("selected"),
                    "history": row.get("history"),
                }
                for row in fold_results
            ] if kfold else None,
            "selected_fold": selected.get("fold") if kfold else None,
            "best_checkpoint": selected.get("best_checkpoint"),
            "best_epoch": selected.get("best_epoch"),
            "best_val_dice": selected.get("best_val_dice"),
            "best_val_loss": selected.get("best_val_loss"),
            "val_metrics": selected.get("val_metrics"),
            "test_metrics": test_metrics,
            "foundation_vs_finetuned": comparison_summary,
            "figures": figure_paths,
            "timing": timing.snapshot_for_status(),
            "selected_checkpoint": {
                "path": selected.get("best_checkpoint"),
                "epoch": selected.get("best_epoch"),
                "fold": selected.get("fold") if kfold else None,
                "criterion": criterion,
                "validation": {
                    "dice": selected.get("best_val_dice"),
                    "loss": selected.get("best_val_loss"),
                },
                "test": test_metrics,
            },
            "progress": 1.0,
            "message": (
                f"Training finished · registered fold {int(selected.get('fold') or 0) + 1} of {n_folds}"
                if kfold else "Training finished"
            ),
        })
        return 0
    finally:
        if timing is not None:
            try:
                timing.flush()
            except Exception:
                pass
        if hydra_cm is not None:
            try:
                hydra_cm.__exit__(None, None, None)
            except Exception:
                pass
        os.chdir(original_cwd)


def _make_optimizer(model, hp, epochs):
    import torch

    param_groups = _optimizer_groups(
        model,
        base_lr=float(hp.get("learning_rate") or 5e-5),
        vision_lr=float(hp.get("vision_learning_rate") or 3e-5),
        weight_decay=float(hp.get("weight_decay") or 0.1),
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = None
    if (hp.get("scheduler") or "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    return optimizer, scheduler


def _restore_cpu(model, snapshot, device):
    import torch

    model.load_state_dict({
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in snapshot.items()
    })


def _load_checkpoint(model, path, device):
    import torch

    payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)


def _train_fold(
    *,
    run_id,
    state,
    trainer,
    optimizer,
    scheduler,
    scaler,
    spec,
    fold_i,
    n_folds,
    epochs,
    device,
    aug,
    rng,
    loss_weights,
    val_every,
    patience,
    use_amp,
    clip_norm,
    num_frames,
    resolution,
    ckpt_dir,
    flags,
    prompt=None,
    hyperparameters=None,
    timing=None,
):
    import torch

    from .storage import STATUS_CANCELLED, cancel_requested, write_run_status

    fold_n = int(spec.get("fold") if spec.get("fold") is not None else fold_i)
    train_examples = spec.get("examples_train") or []
    val_examples = spec.get("examples_validation") or []
    best_dice = None
    best_loss = None
    best_path = None
    best_epoch = None
    stale = 0
    history = []
    prefix = f"Fold {fold_n + 1}/{n_folds} · " if n_folds > 1 else ""

    from .train_batching import (
        GpuBatchPrefetcher,
        forward_loss_from_tensors,
        resolve_training_batch_size,
    )
    from .dataset import preload_example_cache
    from .clip_pipeline import ClipPrefetcher, clip_queue_depth, clip_worker_count
    from .preview_store import (
        assemble_clip_overlay,
        choose_sticky_subject,
        run_preview_dir,
        save_epoch_prediction,
        subject_key,
        update_manifest,
    )

    preview_root = run_preview_dir(run_id)
    preview_payload = {"sticky_key": None, "subjects": [], "epochs": []}

    sample_clip = _first_train_clip(
        train_examples, num_frames, resolution, aug, rng, prompt=prompt,
    )
    hp = hyperparameters or {}
    if timing is not None:
        timing.meta.update({
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "n_train_clips_per_epoch": len(train_examples) * 3,
            "fold": fold_n,
        })
        with timing.span(
            "setup",
            fold=fold_n,
            n_train=len(train_examples),
            n_val=len(val_examples),
        ):
            effective_batch_size = resolve_training_batch_size(
                trainer, device, hp, sample_clip, loss_weights,
            )
            clip_workers = clip_worker_count(hp)
            cached = preload_example_cache(list(train_examples) + list(val_examples))
            val_jobs = _build_eval_clip_jobs(
                val_examples, num_frames, resolution, prompt=prompt,
            )
    else:
        effective_batch_size = resolve_training_batch_size(
            trainer, device, hp, sample_clip, loss_weights,
        )
        clip_workers = clip_worker_count(hp)
        cached = preload_example_cache(list(train_examples) + list(val_examples))
        val_jobs = _build_eval_clip_jobs(
            val_examples, num_frames, resolution, prompt=prompt,
        )
    validation_mode = resolve_validation_mode(hp.get("validation_mode"))
    sticky_volume_example = choose_sticky_subject(val_examples, rng) if val_examples else None
    if sticky_volume_example is not None:
        preview_payload["sticky_key"] = subject_key(sticky_volume_example)
    if timing is not None:
        timing.meta["batch_size"] = effective_batch_size
        timing.meta["clip_workers"] = clip_workers
        timing.meta["n_val_clips"] = len(val_jobs)
        timing.meta["validation_mode"] = validation_mode
    clip_stream = ClipPrefetcher(
        train_examples,
        num_frames=num_frames,
        resolution=resolution,
        aug=aug,
        rng=rng,
        prompt=prompt,
        num_workers=clip_workers,
        queue_depth=clip_queue_depth(effective_batch_size),
        max_epoch=epochs,
    )
    gpu_batches = GpuBatchPrefetcher(device)
    _emit(
        run_id, state,
        phase="train",
        fold=fold_n,
        message=(
            f"{prefix}GPU batch size: {effective_batch_size} clip(s) per step"
            + (f" · preloaded {cached} volume(s)" if cached else "")
            + (f" · {len(val_jobs)} val clips cached" if val_jobs else "")
            + f" · val {validation_mode}"
            + f" · {clip_workers} clip worker(s), streamed"
        ),
        batch_size=effective_batch_size,
    )
    clip_stream.schedule(1)
    clip_stream.schedule(2)

    try:
        for epoch in range(1, epochs + 1):
            if cancel_requested(run_id):
                write_run_status(run_id, {
                    **state,
                    "status": STATUS_CANCELLED,
                    "epoch": epoch,
                    "fold": fold_n,
                    "best_checkpoint": best_path,
                    "best_epoch": best_epoch,
                    "best_val_dice": best_dice,
                    "best_val_loss": best_loss,
                    "message": "Cancelled",
                })
                return {"cancelled": True, "fold": fold_n}

            fold_progress = (fold_i + (epoch - 1) / float(epochs)) / float(n_folds)
            _emit(
                run_id, state,
                phase="train",
                epoch=epoch,
                fold=fold_n,
                progress=fold_progress,
                message=f"{prefix}Training epoch {epoch} of {epochs}",
            )
            trainer.model.train()
            train_loss_count = 0
            n_train_clips = 0
            clip_wait_s = 0.0
            running_loss = torch.zeros((), device=device, dtype=torch.float64)
            clip_stream.schedule(epoch + 1)
            epoch_started = time.perf_counter()

            def _optimizer_step(loss_value):
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss_value).backward()
                    if clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in trainer.model.parameters() if p.requires_grad],
                            clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_value.backward()
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in trainer.model.parameters() if p.requires_grad],
                            clip_norm,
                        )
                    optimizer.step()
                nonlocal train_loss_count
                running_loss.add_(loss_value.detach().double())
                train_loss_count += 1

            waited = time.perf_counter()
            pending = clip_stream.take_batch(epoch, effective_batch_size)
            first_wait_s = time.perf_counter() - waited
            clip_wait_s += first_wait_s
            n_train_clips += len(pending or [])
            if timing is not None:
                timing.event(
                    "first_batch_wait",
                    seconds=round(first_wait_s, 3),
                    fold=fold_n,
                    epoch=epoch,
                    clips=len(pending or []),
                )
            gpu_batches.preload(pending)
            train_started = time.perf_counter()
            while True:
                waited = time.perf_counter()
                nxt = clip_stream.take_batch(epoch, effective_batch_size)
                wait_s = time.perf_counter() - waited
                clip_wait_s += wait_s
                n_train_clips += len(nxt or [])
                if timing is not None and wait_s > 0:
                    timing.event(
                        "clip_wait",
                        seconds=round(wait_s, 3),
                        fold=fold_n,
                        epoch=epoch,
                        clips=len(nxt or []),
                    )
                batch_tensors = gpu_batches.next()
                if batch_tensors is None:
                    break
                gpu_batches.preload(nxt)
                if use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        loss = forward_loss_from_tensors(trainer, batch_tensors, loss_weights)
                else:
                    loss = forward_loss_from_tensors(trainer, batch_tensors, loss_weights)
                _optimizer_step(loss)
                del loss, batch_tensors
            train_s = time.perf_counter() - train_started
            if timing is not None:
                timing.event(
                    "train",
                    seconds=round(train_s, 3),
                    fold=fold_n,
                    epoch=epoch,
                    steps=train_loss_count,
                    clips=n_train_clips,
                )

            train_losses = []
            if train_loss_count:
                train_losses.append(float((running_loss / train_loss_count).item()))

            if scheduler is not None:
                scheduler.step()

            train_dice = None
            val_metrics = None
            val_loss = None
            val_dice = None
            val_clip_train_s = 0.0
            val_clip_val_s = 0.0
            val_volume_s = 0.0
            val_dice_2d = None
            val_dice_3d = None
            dice_source = "clip_2d"
            if val_examples and (epoch % val_every == 0 or epoch == epochs):
                if device.type == "cuda":
                    optimizer.zero_grad(set_to_none=True)
                n_val_clips = len(val_jobs)
                volume_examples = volume_val_examples(
                    validation_mode, val_examples, sticky_volume_example,
                )
                extra = ""
                if volume_examples:
                    extra = (
                        f" · 3D Dice ({len(volume_examples)} volume"
                        f"{'' if len(volume_examples) == 1 else 's'}, 3-view majority vote)"
                    )
                _emit(
                    run_id, state,
                    phase="validate",
                    epoch=epoch,
                    fold=fold_n,
                    train_loss=float(np.mean(train_losses)) if train_losses else None,
                    progress=(fold_i + (epoch - 0.5) / float(epochs)) / float(n_folds),
                    validation_mode=validation_mode,
                    message=(
                        f"{prefix}Validating epoch {epoch} of {epochs}"
                        f" · {n_val_clips} clips, batch {effective_batch_size}"
                        f"{extra}"
                    ),
                )
                started = time.perf_counter()
                clip_val = _evaluate_clip_jobs(
                    trainer,
                    val_jobs,
                    device,
                    batch_size=effective_batch_size,
                    compute_loss=True,
                    use_amp=use_amp,
                    loss_weights=loss_weights,
                    preview_key=(
                        subject_key(sticky_volume_example)
                        if validation_mode == MODE_FAST and sticky_volume_example is not None
                        else None
                    ),
                )
                val_clip_val_s = time.perf_counter() - started
                val_loss = clip_val.get("mean_loss")
                val_metrics = clip_val.get("metrics")
                val_dice_2d = (val_metrics or {}).get("dice")
                val_dice = val_dice_2d
                if timing is not None:
                    timing.event(
                        "val_clip_val",
                        seconds=round(val_clip_val_s, 3),
                        fold=fold_n,
                        epoch=epoch,
                        n=len(val_examples),
                        clips=n_val_clips,
                        batch_size=effective_batch_size,
                    )
                if (
                    validation_mode == MODE_FAST
                    and sticky_volume_example is not None
                    and clip_val.get("clip_previews")
                ):
                    from .dataset import load_example_arrays
                    image, _binary = load_example_arrays(sticky_volume_example)
                    pred = assemble_clip_overlay(image.shape, clip_val["clip_previews"])
                    info = save_epoch_prediction(
                        preview_root,
                        sticky_volume_example,
                        image,
                        pred,
                        epoch=epoch,
                        dice=val_dice_2d,
                    )
                    preview_payload = update_manifest(
                        preview_root,
                        sticky_key=subject_key(sticky_volume_example),
                        subject_info=info,
                        epoch=epoch,
                        preview_kind="clip_slabs",
                    )
                    _emit(run_id, state, preview=preview_payload, epoch=epoch)
                if volume_examples:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    sticky_key = (
                        subject_key(sticky_volume_example)
                        if sticky_volume_example is not None
                        else subject_key(volume_examples[0])
                    )

                    def _save_volume_preview(example, image, pred, metrics):
                        nonlocal preview_payload
                        info = save_epoch_prediction(
                            preview_root,
                            example,
                            image,
                            pred,
                            epoch=epoch,
                            dice=metrics.get("dice"),
                        )
                        preview_payload = update_manifest(
                            preview_root,
                            sticky_key=sticky_key,
                            subject_info=info,
                            epoch=epoch,
                            preview_kind="volume_3d",
                        )
                        _emit(run_id, state, preview=preview_payload, epoch=epoch)

                    started = time.perf_counter()
                    volume_val = _evaluate_volume_split(
                        trainer, volume_examples, device, resolution, prompt=prompt,
                        on_prediction=_save_volume_preview,
                        max_batch=effective_batch_size,
                    )
                    val_volume_s = time.perf_counter() - started
                    vol_metrics = volume_val.get("metrics") or {}
                    val_dice_3d = vol_metrics.get("dice")
                    if val_dice_3d is not None:
                        val_dice = val_dice_3d
                        val_metrics = vol_metrics
                        dice_source = dice_source_for_mode(validation_mode)
                    if timing is not None:
                        timing.event(
                            "val_volume",
                            seconds=round(val_volume_s, 3),
                            fold=fold_n,
                            epoch=epoch,
                            n=len(volume_examples),
                            majority_vote=True,
                            batch_size=effective_batch_size,
                        )

            validated = val_loss is not None
            improved = should_save_checkpoint(val_loss, val_dice, best_loss, best_dice)
            fallback = (not val_examples) and epoch == epochs and best_path is None
            ckpt_s = 0.0
            if improved or fallback:
                if val_loss is not None:
                    best_loss = float(val_loss)
                    # Always replace Dice with this epoch's score. A high Dice from a
                    # worse-loss epoch must not linger into a later plateau comparison.
                    best_dice = None if val_dice is None else float(val_dice)
                best_epoch = epoch
                name = (
                    f"fold{fold_n:02d}_epoch_{epoch:04d}.pt"
                    if n_folds > 1 else f"epoch_{epoch:04d}.pt"
                )
                best_path = str(ckpt_dir / name)
                loss_txt = f"{best_loss:.4f}" if best_loss is not None else "n/a"
                dice_txt = f"{best_dice:.3f}" if best_dice is not None else "n/a"
                _emit(
                    run_id,
                    state,
                    message=f"{prefix}Saving checkpoint for epoch {epoch} (val loss {loss_txt}, Dice {dice_txt})",
                )
                started = time.perf_counter()
                torch.save({"model": trainer.model.state_dict(), "epoch": epoch, "fold": fold_n}, best_path)
                ckpt_s = time.perf_counter() - started
                if timing is not None:
                    timing.event("checkpoint", seconds=round(ckpt_s, 3), fold=fold_n, epoch=epoch)
                stale = 0
            elif validated:
                stale += 1

            epoch_s = time.perf_counter() - epoch_started
            val_s = val_clip_train_s + val_clip_val_s + val_volume_s
            s_per_step = (train_s / train_loss_count) if train_loss_count else None
            s_per_clip = (train_s / n_train_clips) if n_train_clips else None
            s_per_val = (
                (val_clip_val_s / len(val_examples)) if val_examples and val_clip_val_s else None
            )
            if timing is not None:
                timing.event(
                    "epoch",
                    seconds=round(epoch_s, 3),
                    fold=fold_n,
                    epoch=epoch,
                    steps=train_loss_count,
                    clips=n_train_clips,
                    train_s=round(train_s, 3),
                    clip_wait_s=round(clip_wait_s, 3),
                    val_s=round(val_s, 3),
                    val_volume_s=round(val_volume_s, 3),
                    s_per_step=round(s_per_step, 3) if s_per_step is not None else None,
                    s_per_train_clip=round(s_per_clip, 3) if s_per_clip is not None else None,
                    s_per_val_subject=round(s_per_val, 3) if s_per_val is not None else None,
                )
                timing.flush()

            mean_loss = float(np.mean(train_losses)) if train_losses else None
            history.append({
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_loss": val_loss,
                "train_dice": train_dice,
                "val_dice": val_dice,
                "val_dice_2d": val_dice_2d,
                "val_dice_3d": val_dice_3d,
                "val_dice_source": dice_source,
                "validation_mode": validation_mode,
                "val_metrics": val_metrics,
                "checkpoint_saved": bool(improved or fallback),
                "timing": {
                    "epoch_s": round(epoch_s, 3),
                    "train_s": round(train_s, 3),
                    "clip_wait_s": round(clip_wait_s, 3),
                    "val_s": round(val_s, 3),
                    "val_volume_s": round(val_volume_s, 3),
                    "steps": train_loss_count,
                    "clips": n_train_clips,
                },
            })
            parts = [f"{prefix}Epoch {epoch} of {epochs}".strip()]
            if mean_loss is not None:
                parts.append(f"loss {mean_loss:.4f}")
            if val_loss is not None:
                parts.append(f"val loss {val_loss:.4f}")
            if val_dice is not None:
                kind = "3D" if str(dice_source).startswith("volume") else "2D"
                parts.append(f"val Dice {val_dice:.3f} ({kind})")
            if best_loss is not None:
                parts.append(f"best val loss {best_loss:.4f}")
            if best_dice is not None:
                parts.append(f"gated Dice {best_dice:.3f}")
            parts.append(f"{epoch_s:.0f}s ({train_s:.0f}s train / {val_s:.0f}s val)")
            _emit(run_id, state, **{
                "phase": "train",
                "epoch": epoch,
                "epochs": epochs,
                "fold": fold_n,
                "train_loss": mean_loss,
                "val_metrics": val_metrics,
                "best_checkpoint": best_path,
                "best_epoch": best_epoch,
                "best_val_dice": best_dice,
                "best_val_loss": best_loss,
                "validation_mode": validation_mode,
                "progress": (fold_i + epoch / float(epochs)) / float(n_folds),
                "message": " · ".join(parts),
                "history": history,
                "preview": preview_payload,
                "timing": timing.snapshot_for_status() if timing is not None else None,
                "trainable_modules": {
                    name: bool(any(trainable for n, trainable in flags.items() if n.startswith(name)))
                    for name in sorted({n.split(".", 1)[0] for n in flags})
                },
            })
            if patience > 0 and stale >= patience and best_path:
                _emit(run_id, state, message=f"{prefix}Early stop after {stale} epochs without improvement")
                break

    finally:
        clip_stream.close()

    last_val = next((row["val_metrics"] for row in reversed(history) if row.get("val_metrics")), None)
    return {
        "cancelled": False,
        "fold": fold_n,
        "best_checkpoint": best_path,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_val_loss": best_loss,
        "validation_mode": validation_mode,
        "history": history,
        "val_metrics": last_val,
        "examples_validation": val_examples,
        "examples_test": spec.get("examples_test") or [],
        "batch_size": effective_batch_size,
    }


def _first_train_clip(train_examples, num_frames, resolution, aug, rng, prompt=None):
    from .volume_inference import ENSEMBLE_VIEWS

    for example in train_examples:
        for view_index in range(ENSEMBLE_VIEWS):
            clip = _example_to_clip(
                example, num_frames, resolution, aug, rng, prompt=prompt,
                view_index=view_index,
            )
            if clip is not None:
                return clip
    return None


def _example_to_clip(example, num_frames, resolution, aug, rng, prompt=None, view_index=None):
    from .clip_pipeline import example_to_clip

    return example_to_clip(
        example, num_frames, resolution, aug, rng, prompt=prompt, view_index=view_index,
    )


def _tensors_from_clip(clip, device):
    """Contiguous CUDA/CPU tensors — SAM2's .view() cannot take strided batches."""
    import torch
    video = torch.from_numpy(np.ascontiguousarray(clip["video"])).unsqueeze(0).to(device)
    masks = torch.from_numpy(np.ascontiguousarray(clip["masks"])).unsqueeze(0).unsqueeze(2).to(device)
    bbox = torch.from_numpy(np.ascontiguousarray(clip["bbox"])).unsqueeze(0).to(device)
    return video.contiguous(), masks.contiguous(), bbox.contiguous()


def _dense_prompt_tensor(clip, device):
    import torch
    prompt_mask = clip.get("prompt_mask")
    if prompt_mask is None:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(prompt_mask, dtype=np.float32))
    return tensor.unsqueeze(0).unsqueeze(0).to(device).contiguous()


def _forward_loss(trainer, clip, device, weights):
    from .train_batching import forward_loss_from_clips

    return forward_loss_from_clips(trainer, [clip], device, weights)


def _resize_mask_hw(mask2d, height, width):
    from .volume_inference import _resize_mask_hw as resize_mask_hw

    return resize_mask_hw(mask2d, height, width)


def _video_from_z(image_u8, indices, resolution):
    from .volume_inference import _video_from_z as video_from_z

    return video_from_z(image_u8, indices, resolution)


def _infer_prompted_video(trainer, video, bbox, device, dense_mask=None):
    from .volume_inference import _infer_prompted_video as infer_prompted_video

    return infer_prompted_video(trainer, video, bbox, device, dense_mask=dense_mask)


def _propagate_z(trainer, image_u8, indices, prompt_mask, height, width, resolution, device, chunk=8, full_slice=False):
    from .volume_inference import propagate_z

    return propagate_z(
        trainer, image_u8, indices, prompt_mask, height, width, resolution, device,
        chunk=chunk, full_slice=full_slice,
    )


def _predict_binary_volume(trainer, image, binary, device, resolution, prompt=None):
    from .volume_inference import predict_binary_volume

    return predict_binary_volume(trainer, image, binary, device, resolution, prompt=prompt)


def _build_eval_clip_jobs(examples, num_frames, resolution, prompt=None, view_index=None):
    """Deterministic 2D val/test clips. Volumes are loaded once, then reoriented."""
    from .dataset import build_clip, load_example_arrays
    from .volume_inference import ENSEMBLE_VIEWS, reorient_for_view

    view_indices = (
        list(range(ENSEMBLE_VIEWS))
        if view_index is None
        else [int(view_index) % ENSEMBLE_VIEWS]
    )
    jobs = []
    for example in examples or []:
        image, binary = load_example_arrays(example)
        for vi in view_indices:
            view_image = image if vi == 0 else reorient_for_view(image, vi)
            view_binary = binary if vi == 0 else reorient_for_view(binary, vi)
            clip = build_clip(
                view_image, view_binary, num_frames=num_frames, resolution=resolution,
                rng=None, prompt=prompt,
            )
            if clip is None:
                continue
            jobs.append({
                "clip": clip,
                "subject": example.get("subject"),
                "label_id": example.get("label_id"),
                "view_index": vi,
            })
    return jobs


def _evaluate_clip_jobs(
    trainer,
    jobs,
    device,
    *,
    batch_size=1,
    compute_loss=False,
    use_amp=False,
    loss_weights=None,
    collect_previews=False,
    preview_key=None,
):
    """Batched clip eval: one forward per batch for both loss and last-frame Dice."""
    import torch
    from .evaluate import aggregate_metrics
    from .figures import display_frame_from_clip
    from .preview_store import subject_key
    from .train_batching import (
        GpuBatchPrefetcher,
        clip_frames_for_index,
        first_frame_binary,
        last_frame_binary_metrics,
        loss_from_outputs,
        trainer_forward,
    )

    weights = loss_weights or {"loss_mask": 20.0, "loss_dice": 1.0, "loss_iou": 1.0}
    step = max(1, int(batch_size or 1))
    trainer.model.eval()
    rows = []
    loss_sum = 0.0
    loss_count = 0
    prefetcher = GpuBatchPrefetcher(device)
    amp = bool(use_amp) and getattr(device, "type", None) == "cuda"
    want_key = preview_key
    clip_previews = []

    def _clips(chunk):
        return [job["clip"] for job in chunk]

    if jobs:
        prefetcher.preload(_clips(jobs[:step]))
    with torch.no_grad():
        for start in range(0, len(jobs), step):
            chunk = jobs[start:start + step]
            batch_tensors = prefetcher.next()
            nxt = jobs[start + step:start + 2 * step]
            prefetcher.preload(_clips(nxt) if nxt else None)
            if amp:
                with torch.cuda.amp.autocast(enabled=True):
                    pred_masks, pred_logits, pred_ious = trainer_forward(trainer, batch_tensors)
                    loss = loss_from_outputs(pred_logits, pred_ious, batch_tensors[1], weights) if compute_loss else None
            else:
                pred_masks, pred_logits, pred_ious = trainer_forward(trainer, batch_tensors)
                loss = loss_from_outputs(pred_logits, pred_ious, batch_tensors[1], weights) if compute_loss else None
            metrics_rows = last_frame_binary_metrics(pred_masks, batch_tensors[1])
            first_preds = first_frame_binary(pred_masks) if collect_previews else None
            if compute_loss and loss is not None:
                n = max(len(chunk), 1)
                loss_sum += float(loss.detach().cpu()) * n
                loss_count += n
            for index, job in enumerate(chunk):
                row = {
                    "subject": job.get("subject"),
                    "label_id": job.get("label_id"),
                    "view_index": job.get("view_index"),
                    **metrics_rows[index],
                }
                if collect_previews and int(job.get("view_index") or 0) == 0:
                    clip = job["clip"]
                    row["image"] = display_frame_from_clip(clip)
                    row["gt"] = np.asarray(clip["masks"][0]).astype(bool)
                    row["pred"] = first_preds[index]
                if want_key and subject_key({
                    "subject": job.get("subject"),
                    "label_id": job.get("label_id"),
                }) == want_key:
                    clip = job["clip"]
                    clip_previews.append({
                        "subject": job.get("subject"),
                        "label_id": job.get("label_id"),
                        "view_index": job.get("view_index"),
                        "slice_indices": clip.get("slice_indices") or [],
                        "frames": clip_frames_for_index(pred_masks, index),
                    })
                rows.append(row)
            del pred_masks, pred_logits, pred_ious, batch_tensors
    metric_rows = [
        {key: value for key, value in row.items() if key in ("dice", "iou", "hd95")}
        for row in rows
    ]
    payload = {"metrics": aggregate_metrics(metric_rows), "rows": rows}
    if compute_loss and loss_count:
        payload["mean_loss"] = float(loss_sum / loss_count)
    if clip_previews:
        payload["clip_previews"] = clip_previews
    return payload


def _evaluate_clip_split(
    trainer,
    examples,
    device,
    num_frames,
    resolution,
    *,
    prompt=None,
    compute_loss=False,
    view_index=None,
    collect_previews=False,
    batch_size=1,
    use_amp=False,
    loss_weights=None,
):
    """Clip metrics on one or all views (axial / coronal / sagittal)."""
    jobs = _build_eval_clip_jobs(
        examples, num_frames, resolution, prompt=prompt, view_index=view_index,
    )
    return _evaluate_clip_jobs(
        trainer,
        jobs,
        device,
        batch_size=batch_size,
        compute_loss=compute_loss,
        use_amp=use_amp,
        loss_weights=loss_weights,
        collect_previews=collect_previews,
    )


def _evaluate_volume_split(
    trainer, examples, device, resolution, prompt=None, on_prediction=None, max_batch=None,
):
    """Full-volume Dice after 3-view majority vote (axial / coronal / sagittal)."""
    import torch
    from .dataset import load_example_arrays
    from .evaluate import aggregate_metrics, binary_metrics
    from .volume_inference import predict_binary_volume

    trainer.model.eval()
    rows = []
    with torch.no_grad():
        for example in examples:
            image, binary = load_example_arrays(example)
            pred = predict_binary_volume(
                trainer, image, binary, device, resolution, prompt=prompt,
                max_batch=max_batch,
            )
            metrics = binary_metrics(pred, binary, include_hd95=False)
            rows.append({
                "subject": example.get("subject"),
                "label_id": example.get("label_id"),
                **metrics,
            })
            if on_prediction is not None:
                on_prediction(example, image, pred, metrics)
    metric_rows = [
        {key: value for key, value in row.items() if key in ("dice", "iou", "hd95")}
        for row in rows
    ]
    return {"metrics": aggregate_metrics(metric_rows), "rows": rows}


def _evaluate_split(
    trainer,
    examples,
    device,
    num_frames,
    resolution,
    collect_previews=False,
    prompt=None,
    batch_size=1,
    use_amp=False,
    loss_weights=None,
):
    return _evaluate_clip_split(
        trainer, examples, device, num_frames, resolution,
        prompt=prompt, compute_loss=False, view_index=None,
        collect_previews=collect_previews,
        batch_size=batch_size,
        use_amp=use_amp,
        loss_weights=loss_weights,
    )


def _merge_before_after(foundation_rows, finetuned_rows):
    keyed = {
        (row.get("subject"), row.get("label_id")): row
        for row in finetuned_rows
    }
    merged = []
    for row in foundation_rows:
        key = (row.get("subject"), row.get("label_id"))
        other = keyed.get(key) or {}
        merged.append({
            "subject": row.get("subject"),
            "label_id": row.get("label_id"),
            "foundation_dice": row.get("dice"),
            "finetuned_dice": other.get("dice"),
            "foundation_iou": row.get("iou"),
            "finetuned_iou": other.get("iou"),
        })
    return merged


def _optimizer_groups(model, base_lr, vision_lr, weight_decay):
    vision, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("image_encoder."):
            vision.append(parameter)
        else:
            rest.append(parameter)
    groups = []
    if rest:
        groups.append({"params": rest, "lr": base_lr, "weight_decay": weight_decay})
    if vision:
        groups.append({"params": vision, "lr": vision_lr, "weight_decay": weight_decay})
    if not groups:
        raise RuntimeError("No trainable parameters for the selected preset")
    return groups


if __name__ == "__main__":
    sys.exit(main())
