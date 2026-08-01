"""Deterministic patient-level outer and inner fold generation."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.model_selection import StratifiedGroupKFold


def _validated_patients(
    patients: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    rows = sorted(
        (dict(row) for row in patients),
        key=lambda row: str(row["person_key"]),
    )
    keys = [str(row["person_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Each person_key must occur exactly once")
    return rows


def build_outer_folds(
    patients: Iterable[dict[str, object]],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    rows = _validated_patients(patients)
    labels = [str(row["diagnosis"]) for row in rows]
    groups = [str(row["person_key"]) for row in rows]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    assignments = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(rows, labels, groups)
    ):
        for index in test_indices:
            assignments[groups[index]] = fold
    if len(assignments) != len(rows):
        raise ValueError("Outer fold assignment is incomplete")
    return assignments


def build_inner_rows(
    patients: Iterable[dict[str, object]],
    outer_folds: dict[str, int],
    seed: int,
    n_inner_splits: int = 5,
) -> list[dict[str, object]]:
    rows = _validated_patients(patients)
    all_rows = []
    outer_count = max(outer_folds.values()) + 1
    for outer_fold in range(outer_count):
        development = [
            row
            for row in rows
            if outer_folds[str(row["person_key"])] != outer_fold
        ]
        labels = [str(row["diagnosis"]) for row in development]
        groups = [str(row["person_key"]) for row in development]
        splitter = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        _, validation_indices = next(
            splitter.split(development, labels, groups)
        )
        validation_keys = {groups[index] for index in validation_indices}
        for row in rows:
            key = str(row["person_key"])
            if outer_folds[key] == outer_fold:
                split = "test"
            elif key in validation_keys:
                split = "validation"
            else:
                split = "train"
            all_rows.append(
                {
                    "outer_fold": outer_fold,
                    "person_key": key,
                    "diagnosis": str(row["diagnosis"]),
                    "split": split,
                }
            )
    return all_rows
