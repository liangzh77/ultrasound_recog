"""Strict five-fold OOF contract and gate evaluation for G0."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.research_gate import (
    GATE_CLASSES,
    GATE_OOF_COLUMNS,
    bootstrap_roc_auc_ci,
    compute_gate_metrics_from_predictions,
    diagnosis_to_gate_id,
    validate_gate_probabilities,
    write_gate_prediction_csv,
)
from src.research_ledger import sha256_file
from src.research_schema import DIAGNOSIS_CLASSES


@dataclass(frozen=True)
class GateOOFData:
    person_keys: tuple[str, ...]
    targets: np.ndarray
    raw_probabilities: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    outer_folds: np.ndarray
    image_counts: np.ndarray
    thresholds: np.ndarray
    temperatures: np.ndarray
    model_ids: tuple[str, ...]


def _resolve_project_artifact(project_root: Path, value: str) -> Path:
    root = project_root.resolve()
    path = (root / value).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("G0 artifact path escapes the project root") from error
    if relative.startswith("workspace/data/raw/") or "/private/" in f"/{relative}/":
        raise ValueError("G0 public artifact points to raw or private data")
    return path


def _validate_artifact_reference(
    summary: dict[str, Any],
    *,
    path_key: str,
    sha_key: str,
    expected_path: Path | None,
    project_root: Path,
) -> dict[str, Any]:
    path = _resolve_project_artifact(project_root, str(summary[path_key]))
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"G0 summary {path_key} does not match the supplied artifact")
    if not path.is_file():
        raise ValueError(f"G0 summary artifact is missing: {path_key}")
    observed = sha256_file(path)
    if observed != str(summary[sha_key]):
        raise ValueError(f"G0 summary artifact hash mismatch: {path_key}")
    return {
        "path": path.relative_to(project_root.resolve()).as_posix(),
        "sha256": observed,
    }


def _validate_mlflow_runs(database: Path, run_ids: set[str]) -> None:
    if not database.is_file():
        raise ValueError("G0 MLflow database is missing")
    with sqlite3.connect(database) as connection:
        placeholders = ",".join("?" for _ in run_ids)
        rows = connection.execute(
            f"SELECT run_uuid, status FROM runs WHERE run_uuid IN ({placeholders})",
            tuple(sorted(run_ids)),
        ).fetchall()
    observed = {str(run_id): str(status) for run_id, status in rows}
    if set(observed) != run_ids:
        raise ValueError("G0 fold summary references an unknown MLflow run")
    if any(status != "FINISHED" for status in observed.values()):
        raise ValueError("G0 MLflow run is not finished")


def validate_gate_fold_summaries(
    summary_paths: list[Path],
    prediction_paths: list[Path],
    attention_paths: list[Path],
    *,
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Validate five formal folds and every artifact referenced by their summaries."""
    if not (len(summary_paths) == len(prediction_paths) == len(attention_paths) == 5):
        raise ValueError("G0 formal evaluation requires five summaries and fold artifacts")
    expected_config_sha = sha256_file(config_path.resolve())
    summaries: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in summary_paths:
        resolved = path.resolve()
        if resolved.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("G0 fold summary is unexpectedly large")
        summary = json.loads(resolved.read_text(encoding="utf-8"))
        fold = int(summary["outer_fold"])
        if fold in summaries:
            raise ValueError(f"Duplicate G0 fold summary: {fold}")
        summaries[fold] = (resolved, summary)
    if set(summaries) != set(range(5)):
        raise ValueError("G0 summaries must cover folds 0 through 4")

    predictions_by_fold = _paths_by_fold(prediction_paths, "prediction")
    attention_by_fold = _paths_by_fold(attention_paths, "attention")
    expected_seeds = [int(value) for value in config["evaluation"]["seeds"]]
    git_revisions: set[str] = set()
    mlflow_ids: set[str] = set()
    fold_records = []
    database: Path | None = None
    for fold in range(5):
        summary_path, summary = summaries[fold]
        checks = {
            "experiment_code": summary.get("experiment_code") == "G0",
            "task_type": summary.get("task_type") == "binary_normal_abnormal",
            "fold": int(summary.get("outer_fold", -1)) == fold,
            "seed": int(summary.get("seed", -1)) == expected_seeds[fold],
            "formal": summary.get("pilot") is False,
            "clean_git": summary.get("git_dirty") is False,
            "status": summary.get("status") in {"COMPLETED", "EARLY_STOPPED"},
            "outer_test_once": summary.get("outer_test_iterated") is True,
            "no_outer_test_training": summary.get(
                "outer_test_used_for_training_or_early_stopping"
            )
            is False,
            "data_fingerprint": summary.get("data_fingerprint")
            == config["data_fingerprint"],
            "config_sha256": summary.get("config_sha256") == expected_config_sha,
            "pretrained_sha256": summary.get("pretrained_sha256")
            == config["model"]["pretrained_sha256"],
            "threshold_fit_split": summary.get("postprocessing", {})
            .get("operating_threshold", {})
            .get("fit_split")
            == "inner_validation",
            "threshold_constraint": summary.get("postprocessing", {})
            .get("operating_threshold", {})
            .get("constraint_met")
            is True,
            "calibration_fit_split": summary.get("postprocessing", {})
            .get("calibration", {})
            .get("fit_split")
            == "inner_validation",
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"G0 fold {fold} summary contract failed: {failed}")
        git_revision = str(summary.get("git_revision", ""))
        if len(git_revision) != 40:
            raise ValueError(f"G0 fold {fold} has an invalid Git revision")
        git_revisions.add(git_revision)
        fold_mlflow_ids = {
            str(summary.get("mlflow_parent_run_id", "")),
            str(summary.get("mlflow_fold_run_id", "")),
        }
        if any(len(run_id) != 32 for run_id in fold_mlflow_ids):
            raise ValueError(f"G0 fold {fold} has an invalid MLflow run ID")
        if mlflow_ids.intersection(fold_mlflow_ids):
            raise ValueError("G0 formal folds reuse an MLflow run ID")
        mlflow_ids.update(fold_mlflow_ids)
        fold_database = _resolve_project_artifact(
            project_root, str(summary["mlflow_database"])
        )
        if database is None:
            database = fold_database
        elif database != fold_database:
            raise ValueError("G0 fold summaries reference different MLflow databases")

        prediction = _validate_artifact_reference(
            summary,
            path_key="prediction_path",
            sha_key="prediction_sha256",
            expected_path=predictions_by_fold[fold],
            project_root=project_root,
        )
        attention = _validate_artifact_reference(
            summary,
            path_key="attention_path",
            sha_key="attention_sha256",
            expected_path=attention_by_fold[fold],
            project_root=project_root,
        )
        checkpoint = _validate_artifact_reference(
            summary,
            path_key="best_checkpoint_path",
            sha_key="best_checkpoint_sha256",
            expected_path=None,
            project_root=project_root,
        )
        postprocessing = _validate_artifact_reference(
            summary,
            path_key="postprocessing_path",
            sha_key="postprocessing_sha256",
            expected_path=None,
            project_root=project_root,
        )
        elapsed = float(summary["elapsed_hours_total"])
        peak_gpu = float(summary["peak_gpu_memory_reserved_gb"])
        resource_passed = bool(
            elapsed <= float(config["runtime"]["soft_limit_hours"])
            and peak_gpu <= float(config["runtime"]["max_gpu_memory_gb"])
        )
        fold_records.append(
            {
                "fold": fold,
                "seed": expected_seeds[fold],
                "summary_path": summary_path.relative_to(
                    project_root.resolve()
                ).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                "git_revision": git_revision,
                "status": summary["status"],
                "epochs_completed": int(summary["epochs_completed"]),
                "elapsed_hours_total": elapsed,
                "peak_gpu_memory_allocated_gb": float(
                    summary["peak_gpu_memory_allocated_gb"]
                ),
                "peak_gpu_memory_reserved_gb": peak_gpu,
                "gpu": summary["gpu"],
                "mlflow_parent_run_id": summary["mlflow_parent_run_id"],
                "mlflow_fold_run_id": summary["mlflow_fold_run_id"],
                "prediction": prediction,
                "attention": attention,
                "checkpoint": checkpoint,
                "postprocessing": postprocessing,
                "resource_gate_passed": resource_passed,
            }
        )
    if len(git_revisions) != 1:
        raise ValueError("G0 formal folds were not produced from one Git revision")
    assert database is not None
    _validate_mlflow_runs(database, mlflow_ids)
    return {
        "config_path": config_path.resolve().relative_to(project_root.resolve()).as_posix(),
        "config_sha256": expected_config_sha,
        "git_revision": next(iter(git_revisions)),
        "mlflow_database": database.relative_to(project_root.resolve()).as_posix(),
        "folds": fold_records,
        "resource_recording_gate_passed": all(
            row["resource_gate_passed"] for row in fold_records
        ),
    }


def _paths_by_fold(paths: list[Path], kind: str) -> dict[int, Path]:
    mapped: dict[int, Path] = {}
    for path in paths:
        columns, rows = _read_csv(path.resolve())
        if not rows or "outer_fold" not in columns:
            raise ValueError(f"G0 {kind} file is empty or lacks outer_fold")
        folds = {int(row["outer_fold"]) for row in rows}
        if len(folds) != 1:
            raise ValueError(f"Each G0 {kind} file must contain one fold")
        fold = folds.pop()
        if fold in mapped:
            raise ValueError(f"Duplicate G0 {kind} fold: {fold}")
        mapped[fold] = path.resolve()
    if set(mapped) != set(range(5)):
        raise ValueError(f"G0 {kind} files must cover folds 0 through 4")
    return mapped


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def merge_gate_oof_fold_files(fold_paths: list[Path], output_path: Path) -> Path:
    if len(fold_paths) != 5:
        raise ValueError("G0 merge requires exactly five OOF fold files")
    rows: list[dict[str, str]] = []
    seen_folds: set[int] = set()
    seen_people: set[str] = set()
    for path in fold_paths:
        columns, fold_rows = _read_csv(path.resolve())
        if columns != GATE_OOF_COLUMNS or not fold_rows:
            raise ValueError("G0 fold OOF columns or rows do not match the contract")
        folds = {int(row["outer_fold"]) for row in fold_rows}
        if len(folds) != 1:
            raise ValueError("Each G0 OOF file must contain exactly one outer fold")
        fold = folds.pop()
        if fold in seen_folds:
            raise ValueError(f"Duplicate G0 outer fold: {fold}")
        seen_folds.add(fold)
        for row in fold_rows:
            person_key = row["person_key"]
            if person_key in seen_people:
                raise ValueError("Duplicate patient across G0 OOF fold files")
            seen_people.add(person_key)
            rows.append(row)
    if seen_folds != set(range(5)):
        raise ValueError("G0 OOF files must cover outer folds 0 through 4")
    rows.sort(key=lambda row: row["person_key"])
    return write_gate_prediction_csv(output_path, rows)


def _load_registry(registry_dir: Path) -> dict[str, tuple[int, int]]:
    registry_dir = registry_dir.resolve()
    reference = json.loads(
        (registry_dir / "reference_standard.json").read_text(encoding="utf-8")
    )
    if tuple(reference["classes"]) != DIAGNOSIS_CLASSES:
        raise ValueError("G0 registry class order changed")
    _, patient_rows = _read_csv(registry_dir / "patients.csv")
    _, fold_rows = _read_csv(registry_dir / "folds_outer.csv")
    included = {
        row["person_key"]: diagnosis_to_gate_id(row["diagnosis"])
        for row in patient_rows
        if int(row["include"]) == 1
    }
    folds = {row["person_key"]: int(row["outer_fold"]) for row in fold_rows}
    if set(included) != set(folds):
        raise ValueError("G0 included patients and outer folds do not match")
    return {key: (target, folds[key]) for key, target in included.items()}


def load_and_validate_gate_oof(
    prediction_path: Path,
    registry_dir: Path,
    config: dict[str, Any],
) -> GateOOFData:
    columns, rows = _read_csv(prediction_path.resolve())
    if columns != GATE_OOF_COLUMNS or not rows:
        raise ValueError("G0 OOF columns or rows do not match the frozen contract")
    registry = _load_registry(registry_dir)
    person_keys = tuple(row["person_key"] for row in rows)
    if len(person_keys) != len(set(person_keys)) or set(person_keys) != set(registry):
        raise ValueError("G0 OOF must cover every registry patient exactly once")
    targets = np.asarray([int(row["reference_id"]) for row in rows], dtype=np.int64)
    raw_probabilities = validate_gate_probabilities(
        np.asarray(
            [
                [float(row["raw_prob_normal"]), float(row["raw_prob_abnormal"])]
                for row in rows
            ]
        )
    )
    probabilities = validate_gate_probabilities(
        np.asarray(
            [[float(row["prob_normal"]), float(row["prob_abnormal"])] for row in rows]
        )
    )
    predictions = np.asarray([int(row["predicted_id"]) for row in rows], dtype=np.int64)
    outer_folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    image_counts = np.asarray([int(row["image_count"]) for row in rows], dtype=np.int64)
    thresholds = np.asarray(
        [float(row["operating_threshold"]) for row in rows], dtype=np.float64
    )
    temperatures = np.asarray(
        [float(row["temperature"]) for row in rows], dtype=np.float64
    )
    if set(np.unique(targets)) != {0, 1} or not set(np.unique(predictions)) <= {0, 1}:
        raise ValueError("G0 OOF target or prediction IDs are invalid")
    if (image_counts < 1).any() or int(image_counts.sum()) != int(
        config["data"]["expected_images"]
    ):
        raise ValueError("G0 OOF image counts do not match the frozen contract")
    if (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("G0 OOF thresholds must be in [0, 1]")
    if not np.isfinite(temperatures).all() or (temperatures <= 0).any():
        raise ValueError("G0 OOF temperatures must be finite and positive")
    if not np.array_equal(predictions, (probabilities[:, 1] >= thresholds).astype(int)):
        raise ValueError("G0 OOF predictions do not match probabilities and thresholds")

    for index, row in enumerate(rows):
        expected_target, expected_fold = registry[row["person_key"]]
        if targets[index] != expected_target or row["reference_class"] != GATE_CLASSES[expected_target]:
            raise ValueError("G0 OOF reference label differs from the registry")
        if outer_folds[index] != expected_fold:
            raise ValueError("G0 OOF outer fold differs from the registry")
        if row["predicted_class"] != GATE_CLASSES[predictions[index]]:
            raise ValueError("G0 OOF predicted class and ID do not match")
        if row["prediction_level"] != "patient_gate" or not row["model_id"].strip():
            raise ValueError("G0 OOF prediction level or model ID is invalid")

    expected_data = config["data"]
    if len(rows) != int(expected_data["expected_patients"]):
        raise ValueError("G0 OOF patient count does not match the frozen contract")
    if int((targets == 0).sum()) != int(expected_data["expected_normal_patients"]):
        raise ValueError("G0 OOF normal count does not match the frozen contract")
    if int((targets == 1).sum()) != int(expected_data["expected_abnormal_patients"]):
        raise ValueError("G0 OOF abnormal count does not match the frozen contract")
    if set(np.unique(outer_folds)) != set(range(5)):
        raise ValueError("G0 OOF must contain outer folds 0 through 4")
    model_ids = tuple(row["model_id"] for row in rows)
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    for fold in range(5):
        selected = outer_folds == fold
        if set(np.unique(targets[selected])) != {0, 1}:
            raise ValueError("Each G0 fold must contain normal and abnormal patients")
        if len(np.unique(thresholds[selected])) != 1:
            raise ValueError("Each G0 fold must use one operating threshold")
        if len(np.unique(temperatures[selected])) != 1:
            raise ValueError("Each G0 fold must use one temperature")
        if len({model_ids[index] for index in np.flatnonzero(selected)}) != 1:
            raise ValueError("Each G0 fold must use one model ID")
        expected_model_id = f"G0-fold{fold}-seed{seeds[fold]}-formal"
        if {model_ids[index] for index in np.flatnonzero(selected)} != {
            expected_model_id
        }:
            raise ValueError("G0 OOF model ID differs from the frozen formal run")
    return GateOOFData(
        person_keys=person_keys,
        targets=targets,
        raw_probabilities=raw_probabilities,
        probabilities=probabilities,
        predictions=predictions,
        outer_folds=outer_folds,
        image_counts=image_counts,
        thresholds=thresholds,
        temperatures=temperatures,
        model_ids=model_ids,
    )


def validate_gate_attention_alignment(
    attention_rows: list[dict[str, Any]], data: GateOOFData
) -> None:
    expected = {
        person_key: (
            int(data.outer_folds[index]),
            int(data.image_counts[index]),
            data.model_ids[index],
        )
        for index, person_key in enumerate(data.person_keys)
    }
    observed_counts: dict[str, int] = {}
    for row in attention_rows:
        person_key = str(row["person_key"])
        if person_key not in expected:
            raise ValueError("G0 attention contains a patient absent from OOF")
        expected_fold, _, expected_model_id = expected[person_key]
        if int(row["outer_fold"]) != expected_fold:
            raise ValueError("G0 attention fold differs from OOF")
        if row.get("prediction_level") != "image_importance":
            raise ValueError("G0 attention prediction level is invalid")
        if row.get("model_id") != expected_model_id:
            raise ValueError("G0 attention model ID differs from OOF")
        observed_counts[person_key] = observed_counts.get(person_key, 0) + 1
    if set(observed_counts) != set(expected):
        raise ValueError("G0 attention must cover every OOF patient")
    if any(observed_counts[key] != expected[key][1] for key in expected):
        raise ValueError("G0 attention image counts differ from OOF")


def evaluate_gate_oof(
    data: GateOOFData,
    config: dict[str, Any],
    attention_audit: dict[str, Any],
) -> dict[str, Any]:
    metrics = compute_gate_metrics_from_predictions(
        data.targets,
        data.probabilities,
        data.predictions,
    )
    uncalibrated = compute_gate_metrics_from_predictions(
        data.targets,
        data.raw_probabilities,
        data.predictions,
    )
    auc_ci = bootstrap_roc_auc_ci(
        data.targets,
        data.probabilities,
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    fold_metrics = {}
    for fold in range(5):
        selected = data.outer_folds == fold
        fold_metrics[str(fold)] = compute_gate_metrics_from_predictions(
            data.targets[selected],
            data.probabilities[selected],
            data.predictions[selected],
        )
    fold_auc_passes = sum(
        row["roc_auc"] >= 0.75 for row in fold_metrics.values()
    )
    calibrated_not_worse = bool(
        metrics["binary_ece"] <= uncalibrated["binary_ece"] + 1e-12
        and metrics["binary_brier"] <= uncalibrated["binary_brier"] + 1e-12
    )
    pooled_attention = attention_audit["pooled"]
    gates_config = config["gates"]
    gates = {
        "oof_roc_auc": metrics["roc_auc"] >= gates_config["minimum_oof_roc_auc"],
        "oof_roc_auc_ci_low": auc_ci[0] >= gates_config["minimum_oof_roc_auc_ci_low"],
        "abnormal_sensitivity": metrics["abnormal_sensitivity"] >= gates_config["minimum_abnormal_sensitivity"],
        "normal_specificity": metrics["normal_specificity"] >= gates_config["minimum_normal_specificity"],
        "macro_f1": metrics["macro_f1"] >= gates_config["minimum_macro_f1"],
        "fold_auc_stability": fold_auc_passes >= gates_config["minimum_folds_with_roc_auc_at_least_0_75"],
        "calibrated_ece": metrics["binary_ece"] <= gates_config["maximum_calibrated_ece"],
        "calibrated_brier": metrics["binary_brier"] <= gates_config["maximum_calibrated_brier"],
        "calibration_not_worse": calibrated_not_worse,
        "attention_collapse": pooled_attention["multi_image_collapse_rate"] <= gates_config["maximum_multi_image_attention_collapse_rate"],
    }
    return {
        "contract": {
            "prediction_level": "patient_gate",
            "patients": len(data.person_keys),
            "normal_patients": int((data.targets == 0).sum()),
            "abnormal_patients": int((data.targets == 1).sum()),
            "images": int(data.image_counts.sum()),
            "outer_folds": list(range(5)),
            "bootstrap_samples": int(config["evaluation"]["bootstrap_samples"]),
            "bootstrap_seed": int(config["evaluation"]["bootstrap_seed"]),
        },
        "metrics": metrics,
        "uncalibrated_metrics": uncalibrated,
        "roc_auc_95_ci": list(auc_ci),
        "fold_metrics": fold_metrics,
        "folds_with_roc_auc_at_least_0_75": int(fold_auc_passes),
        "thresholds_by_fold": {
            str(fold): float(np.unique(data.thresholds[data.outer_folds == fold])[0])
            for fold in range(5)
        },
        "temperatures_by_fold": {
            str(fold): float(np.unique(data.temperatures[data.outer_folds == fold])[0])
            for fold in range(5)
        },
        "attention": attention_audit,
        "gates": gates,
        "performance_attention_gate_passed": all(gates.values()),
    }
