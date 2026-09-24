"""Append-only timing log for a fine-tuning run.

Writes:
- ``logs/timing.jsonl`` — every span (crash-safe)
- ``logs/timing.json`` — rolling summary with percentages
- ``logs/timing.txt`` — human-readable table
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .storage import atomic_write_json, logs_dir

SPAN_NAMES = (
    "setup",
    "first_batch_wait",
    "clip_wait",
    "train",
    "val_clip_train",
    "val_clip_val",
    "val_volume",
    "checkpoint",
    "compare",
    "test",
    "epoch",
)


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


class TimingLog:
    def __init__(self, run_id: str, extra: Optional[Dict[str, Any]] = None):
        self.run_id = str(run_id)
        self.dir = logs_dir(run_id)
        self.jsonl_path = self.dir / "timing.jsonl"
        self.json_path = self.dir / "timing.json"
        self.txt_path = self.dir / "timing.txt"
        self.t0 = time.perf_counter()
        self.events: List[Dict[str, Any]] = []
        self.meta = dict(extra or {})
        self._write_event({
            "name": "run_start",
            "t": 0.0,
            **self.meta,
        })

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def event(self, name: str, **fields: Any) -> Dict[str, Any]:
        payload = {"name": str(name), "t": _round(self.elapsed()), **fields}
        self._write_event(payload)
        return payload

    @contextmanager
    def span(self, name: str, **fields: Any):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.event(name, seconds=_round(time.perf_counter() - started), **fields)

    def summary(self) -> Dict[str, Any]:
        totals: Dict[str, float] = {}
        epochs: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for event in self.events:
            name = event.get("name")
            seconds = event.get("seconds")
            if seconds is None:
                continue
            if name in SPAN_NAMES:
                totals[name] = totals.get(name, 0.0) + float(seconds)
            if name == "epoch":
                row = {
                    "fold": event.get("fold"),
                    "epoch": event.get("epoch"),
                    "seconds": float(seconds),
                    "steps": event.get("steps"),
                    "clips": event.get("clips"),
                    "train_s": event.get("train_s"),
                    "clip_wait_s": event.get("clip_wait_s"),
                    "val_s": event.get("val_s"),
                    "val_volume_s": event.get("val_volume_s"),
                    "s_per_step": event.get("s_per_step"),
                    "s_per_train_clip": event.get("s_per_train_clip"),
                    "s_per_val_subject": event.get("s_per_val_subject"),
                }
                epochs.append(row)
                current = row
        work_keys = [key for key in SPAN_NAMES if key != "epoch"]
        work_total = sum(totals.get(key, 0.0) for key in work_keys) or 1.0
        share = {
            key: _round(100.0 * totals.get(key, 0.0) / work_total, 1)
            for key in work_keys
            if totals.get(key, 0.0) > 0
        }
        val_s = (
            totals.get("val_clip_train", 0.0)
            + totals.get("val_clip_val", 0.0)
            + totals.get("val_volume", 0.0)
        )
        payload = {
            "run_id": self.run_id,
            "elapsed_s": _round(self.elapsed()),
            "meta": self.meta,
            "totals_s": {key: _round(value) for key, value in totals.items()},
            "share_pct": share,
            "val_total_s": _round(val_s),
            "train_total_s": _round(totals.get("train", 0.0)),
            "last_epoch": current or None,
            "epochs": epochs,
            "verdict": _verdict(totals, val_s, work_total),
            "files": {
                "jsonl": str(self.jsonl_path),
                "json": str(self.json_path),
                "txt": str(self.txt_path),
            },
        }
        return payload

    def flush(self) -> Dict[str, Any]:
        payload = self.summary()
        atomic_write_json(self.json_path, payload)
        self.txt_path.write_text(_format_txt(payload), encoding="utf-8")
        return payload

    def snapshot_for_status(self) -> Dict[str, Any]:
        payload = self.summary()
        last = payload.get("last_epoch") or {}
        return {
            "elapsed_s": payload.get("elapsed_s"),
            "train_total_s": payload.get("train_total_s"),
            "val_total_s": payload.get("val_total_s"),
            "share_pct": payload.get("share_pct"),
            "last_epoch": last or None,
            "verdict": payload.get("verdict"),
            "log": "logs/timing.txt",
        }

    def _write_event(self, payload: Dict[str, Any]) -> None:
        self.events.append(payload)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def _verdict(totals: Dict[str, float], val_s: float, work_total: float) -> str:
    train_s = totals.get("train", 0.0)
    volume_s = totals.get("val_volume", 0.0)
    clip_wait = totals.get("clip_wait", 0.0) + totals.get("first_batch_wait", 0.0)
    if work_total <= 0:
        return "No timed work yet."
    val_pct = 100.0 * val_s / work_total
    train_pct = 100.0 * train_s / work_total
    wait_pct = 100.0 * clip_wait / max(train_s + clip_wait, 1e-6)
    if val_pct >= 45:
        extra = ""
        if volume_s >= 0.6 * max(val_s, 1e-6):
            extra = (
                " Most of that is 3-view volume Dice, which scales almost linearly "
                "with the number of validation subjects."
            )
        return (
            f"Validation is {val_pct:.0f}% of timed work ({val_s:.1f}s vs train {train_s:.1f}s)."
            + extra
        )
    if wait_pct >= 20:
        return (
            f"GPU is waiting on CPU clip prep for {wait_pct:.0f}% of the train loop. "
            "Clip streaming is not keeping the GPU fully fed."
        )
    if train_pct >= 55:
        return (
            f"Training steps are {train_pct:.0f}% of timed work and clip-wait is low. "
            "This is close to peak speed for the current batch size and MedSAM2 sequential frames."
        )
    return (
        f"Train {train_pct:.0f}% / val {val_pct:.0f}% of timed work. "
        "See logs/timing.txt for the per-epoch breakdown."
    )


def _format_txt(payload: Dict[str, Any]) -> str:
    lines = [
        f"Fine-tune timing  run {payload.get('run_id')}",
        f"Elapsed {payload.get('elapsed_s')} s",
        "",
        payload.get("verdict") or "",
        "",
        "Totals (seconds)",
    ]
    totals = payload.get("totals_s") or {}
    share = payload.get("share_pct") or {}
    for key, value in totals.items():
        pct = share.get(key)
        suffix = f"  ({pct}%)" if pct is not None else ""
        lines.append(f"  {key:<18} {value:.3f}{suffix}")
    lines.append("")
    lines.append("Per epoch")
    lines.append(
        "  fold  ep   epoch_s  train_s  wait_s  val_s  vol_s  steps  clips  s/step  s/clip  s/val"
    )
    for row in payload.get("epochs") or []:
        lines.append(
            "  {fold:>4} {epoch:>3} {seconds:>8.1f} {train_s:>8.1f} {clip_wait_s:>7.1f} "
            "{val_s:>6.1f} {val_volume_s:>6.1f} {steps:>6} {clips:>6} {s_per_step:>7} "
            "{s_per_train_clip:>7} {s_per_val_subject:>6}".format(
                fold=row.get("fold") if row.get("fold") is not None else "-",
                epoch=row.get("epoch") if row.get("epoch") is not None else "-",
                seconds=float(row.get("seconds") or 0),
                train_s=float(row.get("train_s") or 0),
                clip_wait_s=float(row.get("clip_wait_s") or 0),
                val_s=float(row.get("val_s") or 0),
                val_volume_s=float(row.get("val_volume_s") or 0),
                steps=row.get("steps") if row.get("steps") is not None else "-",
                clips=row.get("clips") if row.get("clips") is not None else "-",
                s_per_step=_fmt(row.get("s_per_step")),
                s_per_train_clip=_fmt(row.get("s_per_train_clip")),
                s_per_val_subject=_fmt(row.get("s_per_val_subject")),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"
