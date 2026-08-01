"""Load and evaluate the study's deidentified patient-level OOF contract."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
    validate_oof_predictions,
)
from src.research_schema import DIAGNOSIS_CLASSES


PROBABILITY_COLUMNS = (
    "prob_normal",
    "prob_ra",
    "prob_ga",
    "prob_spa",
    "prob_oa",
    "prob_injury",
)
REQUIRED_COLUMNS = (
    "prediction_level",
    "person_key",
    "outer_fold",
    "reference_class",
    "reference_id",
    *PROBABILITY_COLUMNS,
    "image_count",
    "model_id",
)


@dataclass(frozen=True)
class OOFData:
    person_keys: tuple[str, ...]
    targets: np.ndarray
    probabilities: np.ndarray
    outer_folds: np.ndarray
    image_counts: np.ndarray
    model_ids: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(registry_dir: Path) -> dict[str, tuple[int, int, str]]:
    registry_dir = registry_dir.resolve()
    reference = json.loads(
        (registry_dir / "reference_standard.json").read_text(encoding="utf-8")
    )
    if tuple(reference["classes"]) != DIAGNOSIS_CLASSES:
        raise ValueError("Registry class order does not match the frozen class contract")

    included = {
        row["person_key"]: (int(row["diagnosis_id"]), row["diagnosis"])
        for row in _read_csv(registry_dir / "patients.csv")
        if int(row["include"]) == 1
    }
    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in _read_csv(registry_dir / "folds_outer.csv")
    }
    if set(included) != set(folds):
        raise ValueError("Included patients and outer-fold registry do not match")
    return {
        person_key: (diagnosis_id, folds[person_key], diagnosis)
        for person_key, (diagnosis_id, diagnosis) in included.items()
    }


def _load_and_validate(prediction_path: Path, registry_dir: Path) -> OOFData:
    rows = _read_csv(prediction_path.resolve())
    if not rows:
        raise ValueError("OOF prediction file is empty")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(rows[0]))
    if missing_columns:
        raise ValueError(f"Missing OOF columns: {', '.join(missing_columns)}")

    registry = _load_registry(registry_dir)
    person_keys = tuple(row["person_key"] for row in rows)
    targets = np.asarray([int(row["reference_id"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    outer_folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    image_counts = np.asarray([int(row["image_count"]) for row in rows], dtype=np.int64)
    levels = {row["prediction_level"] for row in rows}
    if levels != {"patient"}:
        raise ValueError("OOF predictions must contain only patient-level rows")
    validate_oof_predictions(
        person_keys,
        targets,
        probabilities,
        expected_person_keys=registry,
        prediction_level="patient",
    )
    if (image_counts < 1).any():
        raise ValueError("Every patient prediction must include at least one image")

    for index, row in enumerate(rows):
        expected_id, expected_fold, expected_class = registry[row["person_key"]]
        if targets[index] != expected_id or row["reference_class"] != expected_class:
            raise ValueError(f"reference diagnosis mismatch for {row['person_key']}")
        if outer_folds[index] != expected_fold:
            raise ValueError(f"outer_fold mismatch for {row['person_key']}")
        if not row["model_id"].strip():
            raise ValueError("model_id cannot be empty")

    return OOFData(
        person_keys=person_keys,
        targets=targets,
        probabilities=probabilities,
        outer_folds=outer_folds,
        image_counts=image_counts,
        model_ids=tuple(row["model_id"] for row in rows),
    )


def evaluate_oof_file(
    prediction_path: Path,
    registry_dir: Path,
    n_bootstrap: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    data = _load_and_validate(prediction_path, registry_dir)
    interval = bootstrap_macro_f1_ci(
        data.targets,
        data.probabilities,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    fold_metrics = {
        str(fold): compute_patient_metrics(
            data.targets[data.outer_folds == fold],
            data.probabilities[data.outer_folds == fold],
            DIAGNOSIS_CLASSES,
        )
        for fold in sorted(np.unique(data.outer_folds))
    }
    summary_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auc",
        "top2_accuracy",
        "multiclass_brier",
        "ece",
    )
    fold_summary = {}
    for name in summary_names:
        values = [fold_metrics[fold][name] for fold in fold_metrics]
        finite_values = [value for value in values if value is not None]
        fold_summary[name] = {
            "mean": float(np.mean(finite_values)) if finite_values else None,
            "standard_deviation": (
                float(np.std(finite_values)) if finite_values else None
            ),
        }
    return {
        "contract": {
            "prediction_level": "patient",
            "patients": len(data.person_keys),
            "unique_patients": len(set(data.person_keys)),
            "classes": list(DIAGNOSIS_CLASSES),
            "outer_folds": sorted(int(value) for value in np.unique(data.outer_folds)),
            "probability_sum_tolerance": 1e-6,
            "bootstrap_samples": n_bootstrap,
            "bootstrap_seed": seed,
        },
        "prediction_sha256": _sha256(prediction_path),
        "model_ids": sorted(set(data.model_ids)),
        "metrics": compute_patient_metrics(
            data.targets,
            data.probabilities,
            DIAGNOSIS_CLASSES,
        ),
        "macro_f1_95_ci": list(interval),
        "fold_metrics": fold_metrics,
        "fold_summary": fold_summary,
    }


def compare_oof_files(
    baseline_path: Path,
    candidate_path: Path,
    registry_dir: Path,
    n_bootstrap: int = 2000,
    seed: int = 20260724,
) -> dict[str, Any]:
    baseline = _load_and_validate(baseline_path, registry_dir)
    candidate = _load_and_validate(candidate_path, registry_dir)
    if set(baseline.person_keys) != set(candidate.person_keys):
        raise ValueError("Paired OOF files must cover identical patients")
    baseline_index = {key: index for index, key in enumerate(baseline.person_keys)}
    order = np.asarray([baseline_index[key] for key in candidate.person_keys])
    if not np.array_equal(baseline.targets[order], candidate.targets):
        raise ValueError("Paired OOF files have different reference diagnoses")
    return {
        "comparison": "candidate_minus_baseline_macro_f1",
        "baseline_sha256": _sha256(baseline_path),
        "candidate_sha256": _sha256(candidate_path),
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": seed,
        **paired_bootstrap_macro_f1(
            candidate.targets,
            baseline.probabilities[order],
            candidate.probabilities,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }
