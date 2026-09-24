"""Overlapped clip preparation so the GPU stays fed across epoch and validation boundaries."""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Full, Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .volume_inference import ENSEMBLE_VIEWS

BuildFn = Callable[..., Optional[Dict[str, Any]]]


def clip_worker_count(hyperparameters: Optional[Dict[str, Any]] = None) -> int:
    """CPU workers for clip production (0 in UI → sensible default)."""
    hp = hyperparameters or {}
    requested = int(hp.get("num_workers") or 0)
    if requested > 0:
        return min(requested, 8)
    cpus = os.cpu_count() or 4
    return min(4, max(2, cpus // 2))


def clip_queue_depth(batch_size: int) -> int:
    """How many finished clips to keep ready (current batch + next + lookahead)."""
    return max(int(batch_size) * 4, 8)


def example_to_clip(
    example,
    num_frames,
    resolution,
    aug,
    rng,
    prompt=None,
    view_index=None,
):
    from .dataset import augment_clip, build_clip, load_example_arrays
    from .volume_inference import reorient_for_view

    image, binary = load_example_arrays(example)
    if view_index is None:
        view_index = int(rng.integers(0, ENSEMBLE_VIEWS)) if rng is not None else 0
    else:
        view_index = int(view_index) % ENSEMBLE_VIEWS
    if view_index != 0:
        image = reorient_for_view(image, view_index)
        binary = reorient_for_view(binary, view_index)
    clip = build_clip(
        image, binary, num_frames=num_frames, resolution=resolution, rng=rng, prompt=prompt,
    )
    if clip is None:
        return None
    clip["view_index"] = view_index
    return augment_clip(clip, aug, rng)


def _build_one_clip(
    example,
    view_index,
    num_frames,
    resolution,
    aug,
    prompt,
    seed,
):
    rng = np.random.default_rng(int(seed))
    return example_to_clip(
        example,
        num_frames,
        resolution,
        aug,
        rng,
        prompt=prompt,
        view_index=view_index,
    )


def epoch_clip_jobs(
    train_examples: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Tuple[Any, int, int]]:
    """Shuffled (example, view, seed) jobs for one epoch."""
    jobs: List[Tuple[Any, int, int]] = []
    base_seed = int(rng.integers(0, 2**31 - 1))
    job_index = 0
    for example in train_examples:
        for view_index in range(ENSEMBLE_VIEWS):
            jobs.append((example, view_index, base_seed + job_index))
            job_index += 1
    rng.shuffle(jobs)
    return jobs


class ClipPrefetcher:
    """CPU producer that stays one epoch ahead of the GPU consumer.

    Workers keep building clips into a bounded queue while the GPU trains.
    During validation the same workers fill the *next* epoch, so training
    resumes without a clip-assembly stall.
    """

    def __init__(
        self,
        train_examples: Sequence[Dict[str, Any]],
        *,
        num_frames: int,
        resolution: int,
        aug,
        rng: np.random.Generator,
        prompt=None,
        num_workers: int = 4,
        queue_depth: int = 8,
        max_epoch: int = 1,
        build_fn: Optional[BuildFn] = None,
    ):
        self._examples = list(train_examples)
        self._num_frames = int(num_frames)
        self._resolution = int(resolution)
        self._aug = aug
        self._rng = rng
        self._prompt = prompt
        self._build_fn = build_fn or _build_one_clip
        self._max_epoch = max(1, int(max_epoch))
        self._workers = max(1, min(int(num_workers), 8))
        self._out_q: Queue = Queue(maxsize=max(1, int(queue_depth)))
        self._hold: Dict[int, deque] = defaultdict(deque)
        self._expected: Dict[int, int] = {}
        self._received: Dict[int, int] = defaultdict(int)
        self._scheduled = set()
        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._closed = False
        self._pool: Optional[ThreadPoolExecutor] = None
        if self._examples:
            self._pool = ThreadPoolExecutor(max_workers=self._workers)

    def schedule(self, epoch: int) -> None:
        """Submit jobs for ``epoch`` if it is in range and not already queued."""
        epoch = int(epoch)
        if epoch < 1 or epoch > self._max_epoch or not self._examples or self._pool is None:
            return
        with self._lock:
            if epoch in self._scheduled or self._closed:
                return
            self._scheduled.add(epoch)
            jobs = epoch_clip_jobs(self._examples, self._rng)
            self._expected[epoch] = len(jobs)
        for example, view_index, seed in jobs:
            self._pool.submit(self._run_job, epoch, example, view_index, seed)

    def take_batch(self, epoch: int, batch_size: int) -> List[Dict[str, Any]]:
        """Return up to ``batch_size`` clips for ``epoch`` (empty when that epoch is done)."""
        self.schedule(epoch)
        self.schedule(epoch + 1)
        batch: List[Dict[str, Any]] = []
        while len(batch) < max(1, int(batch_size)) and not self._epoch_done(epoch):
            clip = self._next_item(epoch)
            if clip is not None:
                batch.append(clip)
        return batch

    def close(self) -> None:
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _epoch_done(self, epoch: int) -> bool:
        expected = self._expected.get(int(epoch))
        if expected is None:
            return True
        return self._received[int(epoch)] >= expected

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Clip prefetcher failed") from self._error

    def _run_job(self, epoch, example, view_index, seed) -> None:
        try:
            clip = self._build_fn(
                example,
                view_index,
                self._num_frames,
                self._resolution,
                self._aug,
                self._prompt,
                seed,
            )
            self._put_out(int(epoch), clip, None)
        except BaseException as exc:
            self._error = exc
            self._put_out(int(epoch), None, exc)

    def _put_out(self, epoch, clip, error) -> None:
        while not self._closed:
            try:
                self._out_q.put((epoch, clip, error), timeout=0.2)
                return
            except Full:
                continue

    def _next_item(self, epoch: int):
        epoch = int(epoch)
        while True:
            self._raise_if_failed()
            held = self._hold[epoch]
            if held:
                self._received[epoch] += 1
                return held.popleft()
            tagged_epoch, clip, error = self._out_q.get()
            if error is not None:
                self._error = error
                self._raise_if_failed()
            if tagged_epoch != epoch:
                self._hold[tagged_epoch].append(clip)
                continue
            self._received[epoch] += 1
            return clip


def prebuild_epoch_clips(
    train_examples: Sequence[Dict[str, Any]],
    *,
    num_frames: int,
    resolution: int,
    aug,
    rng: np.random.Generator,
    prompt=None,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Build every (subject × view) clip for one epoch (tests / fallback)."""
    prefetcher = ClipPrefetcher(
        train_examples,
        num_frames=num_frames,
        resolution=resolution,
        aug=aug,
        rng=rng,
        prompt=prompt,
        num_workers=num_workers,
        queue_depth=max(len(train_examples) * ENSEMBLE_VIEWS, 1),
        max_epoch=1,
    )
    try:
        prefetcher.schedule(1)
        clips: List[Dict[str, Any]] = []
        while True:
            batch = prefetcher.take_batch(1, 32)
            if not batch:
                break
            clips.extend(batch)
        return clips
    finally:
        prefetcher.close()
