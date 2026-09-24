"""Subject-level train / validation / test splits. Never split by slice."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SplitAssignments = Dict[str, List[str]]


class SplitError(ValueError):
    pass


def subject_level_split(
    subjects: Sequence[str],
    *,
    train_pct: float = 70,
    val_pct: float = 15,
    test_pct: float = 15,
    seed: int = 123,
    allow_no_test: bool = False,
) -> SplitAssignments:
    """Reproducible subject-level partition.

    Percentages are of the subject count. Every selected subject appears in
    exactly one of train / validation / test.
    """
    names = _unique_subjects(subjects)
    n = len(names)
    if n == 0:
        raise SplitError("No subjects provided for the split")

    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 100.0) > 1e-6:
        raise SplitError(f"Split percentages must sum to 100, got {total_pct}")

    if test_pct <= 0 and not allow_no_test:
        raise SplitError("Test percentage must be positive unless allow_no_test is set")

    if n < 2:
        raise SplitError("Need at least 2 subjects for a train/validation split")

    rng = random.Random(int(seed))
    shuffled = list(names)
    rng.shuffle(shuffled)

    n_test = int(round(n * test_pct / 100.0)) if test_pct > 0 else 0
    n_val = int(round(n * val_pct / 100.0))
    if test_pct > 0 and n_test < 1:
        n_test = 1
    if n_val < 1:
        n_val = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        raise SplitError(
            f"Not enough subjects ({n}) for the requested split "
            f"{train_pct}/{val_pct}/{test_pct}"
        )

    train = shuffled[:n_train]
    validation = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    assignments = {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
    }
    assert_no_subject_leakage(assignments)
    return assignments


def k_fold_splits(
    subjects: Sequence[str],
    *,
    k: int,
    seed: int = 123,
    holdout_test: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Subject-level k-fold CV, optionally with a fixed held-out test set.

    Fold subjects are never mixed with the held-out test subjects.
    """
    if k < 2:
        raise SplitError("k-fold cross-validation requires k >= 2")

    holdout = set(_unique_subjects(holdout_test or []))
    pool = [name for name in _unique_subjects(subjects) if name not in holdout]
    if len(pool) < k:
        raise SplitError(
            f"Need at least {k} subjects for {k}-fold CV, got {len(pool)}"
        )

    rng = random.Random(int(seed))
    shuffled = list(pool)
    rng.shuffle(shuffled)

    folds: List[List[str]] = [[] for _ in range(k)]
    for index, name in enumerate(shuffled):
        folds[index % k].append(name)

    results = []
    for fold_index in range(k):
        validation = sorted(folds[fold_index])
        train = sorted(
            name
            for other_index, other in enumerate(folds)
            if other_index != fold_index
            for name in other
        )
        assignments = {
            "train": train,
            "validation": validation,
            "test": sorted(holdout),
            "fold": fold_index,
        }
        assert_no_subject_leakage(
            {key: assignments[key] for key in ("train", "validation", "test")}
        )
        results.append(assignments)
    return results


def assert_no_subject_leakage(assignments: Mapping[str, Iterable[str]]) -> None:
    seen = {}
    for split_name, names in assignments.items():
        if split_name == "fold":
            continue
        for name in names:
            if name in seen:
                raise SplitError(
                    f"Subject {name!r} appears in both {seen[name]!r} and {split_name!r}"
                )
            seen[name] = split_name


def assignments_from_user(
    train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str],
) -> SplitAssignments:
    assignments = {
        "train": _unique_subjects(train),
        "validation": _unique_subjects(validation),
        "test": _unique_subjects(test),
    }
    if not assignments["train"]:
        raise SplitError("Train split is empty")
    if not assignments["validation"]:
        raise SplitError("Validation split is empty")
    assert_no_subject_leakage(assignments)
    return assignments


def _unique_subjects(subjects: Optional[Iterable[str]]) -> List[str]:
    seen = []
    used = set()
    for name in subjects or []:
        text = str(name).strip()
        if not text or text in used:
            continue
        used.add(text)
        seen.append(text)
    return seen
