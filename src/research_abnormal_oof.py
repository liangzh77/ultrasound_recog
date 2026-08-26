"""Formal OOF evaluation contract for the D0 abnormal five-class model."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.research_abnormal import D0_CLASS_SLUGS, D0_PROBABILITY_COLUMNS
from src.research_attention_audit import audit_attention_rows, read_attention_files
from src.research_clinical import CLINICAL_CLASSES
from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
    validate_oof_predictions,
)


BASELINE_PROBABILITY_COLUMNS = tuple(
    f"e2_conditional_prob_{slug}" for slug in D0_CLASS_SLUGS
)
D0_REQUIRED_COLUMNS = {
    "prediction_level",
    "person_key",
    "outer_fold",
    "reference_class",
    "reference_id",
    "image_count",
    "model_id",
    *D0_PROBABILITY_COLUMNS,
}
BASELINE_REQUIRED_COLUMNS = {
    "person_key",
    "outer_fold",
    "reference_class",
    "reference_id",
    *BASELINE_PROBABILITY_COLUMNS,
}


@dataclass(frozen=True)
class D0OOFData:
    rows: tuple[dict[str, str], ...]
    person_keys: tuple[str, ...]
    outer_folds: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"OOF file is empty: {path}")
    return rows


def _probabilities(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str]
) -> np.ndarray:
    try:
        result = np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OOF probability columns are invalid") from error
    return result


def _index_unique(
    rows: Sequence[dict[str, str]], label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("person_key", "").strip()
        if not key or key in result:
            raise ValueError(f"{label} OOF person keys must be nonempty and unique")
        result[key] = row
    return result


def load_d0_oof_inputs(
    fold_paths: Sequence[Path], baseline_path: Path, config: Mapping[str, Any]
) -> tuple[D0OOFData, D0OOFData, list[dict[str, Any]]]:
    expected_folds = tuple(int(value) for value in config["evaluation"]["outer_folds"])
    expected_seeds = tuple(int(value) for value in config["evaluation"]["seeds"])
    if len(fold_paths) != len(expected_folds):
        raise ValueError("D0 requires exactly one OOF file for each frozen outer fold")
    if sha256_file(baseline_path) != str(config["evaluation"]["baseline"]["sha256"]):
        raise ValueError("D0 baseline SHA-256 contract failed")

    combined: list[dict[str, str]] = []
    inputs: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    seen_folds: set[int] = set()
    for path in fold_paths:
        rows = _read_csv(path)
        missing = D0_REQUIRED_COLUMNS.difference(rows[0])
        if missing:
            raise ValueError(f"D0 OOF is missing columns: {sorted(missing)}")
        current_fields = list(rows[0])
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise ValueError("D0 fold files use different columns")
        folds = {int(row["outer_fold"]) for row in rows}
        if len(folds) != 1:
            raise ValueError("Each D0 OOF file must contain exactly one fold")
        fold = folds.pop()
        if fold in seen_folds or fold not in expected_folds:
            raise ValueError("D0 OOF contains a duplicate or unexpected fold")
        seed = expected_seeds[expected_folds.index(fold)]
        expected_model = f"D0-fold{fold}-seed{seed}-formal"
        if {row["model_id"] for row in rows} != {expected_model}:
            raise ValueError("D0 OOF model identity differs from the frozen contract")
        if {row["prediction_level"] for row in rows} != {"patient"}:
            raise ValueError("D0 OOF must be patient-level")
        seen_folds.add(fold)
        combined.extend(rows)
        inputs.append(
            {"fold": fold, "path": path.as_posix(), "sha256": sha256_file(path), "rows": len(rows)}
        )
    if seen_folds != set(expected_folds):
        raise ValueError("D0 OOF fold coverage differs from the frozen contract")

    baseline_rows = _read_csv(baseline_path)
    missing = BASELINE_REQUIRED_COLUMNS.difference(baseline_rows[0])
    if missing:
        raise ValueError(f"D0 baseline is missing columns: {sorted(missing)}")
    candidate_index = _index_unique(combined, "D0")
    baseline_index = _index_unique(baseline_rows, "baseline")
    expected_patients = int(config["data"]["expected_patients"])
    if (
        len(candidate_index) != expected_patients
        or len(baseline_index) != expected_patients
        or set(candidate_index) != set(baseline_index)
    ):
        raise ValueError("D0 and baseline OOF coverage differs from the frozen cohort")

    keys = tuple(sorted(candidate_index))
    candidates = tuple(candidate_index[key] for key in keys)
    baselines = tuple(baseline_index[key] for key in keys)
    targets = np.asarray([int(row["reference_id"]) for row in candidates], dtype=np.int64)
    folds = np.asarray([int(row["outer_fold"]) for row in candidates], dtype=np.int64)
    candidate_probabilities = _probabilities(candidates, D0_PROBABILITY_COLUMNS)
    baseline_probabilities = _probabilities(baselines, BASELINE_PROBABILITY_COLUMNS)
    validate_oof_predictions(keys, targets, candidate_probabilities, keys, "patient")
    validate_oof_predictions(keys, targets, baseline_probabilities, keys, "patient")

    for candidate, baseline in zip(candidates, baselines):
        target = int(candidate["reference_id"])
        if (
            int(baseline["reference_id"]) != target
            or int(baseline["outer_fold"]) != int(candidate["outer_fold"])
            or baseline["reference_class"] != candidate["reference_class"]
            or candidate["reference_class"] != CLINICAL_CLASSES[target]
            or int(candidate["image_count"]) < 1
        ):
            raise ValueError("D0 paired reference, fold, or image-count contract failed")

    return (
        D0OOFData(candidates, keys, folds, targets, candidate_probabilities),
        D0OOFData(baselines, keys, folds.copy(), targets.copy(), baseline_probabilities),
        sorted(inputs, key=lambda item: item["fold"]),
    )


def _model_report(data: D0OOFData, samples: int, seed: int) -> dict[str, Any]:
    fold_metrics = {
        str(fold): compute_patient_metrics(
            data.targets[data.outer_folds == fold],
            data.probabilities[data.outer_folds == fold],
            CLINICAL_CLASSES,
        )
        for fold in sorted(np.unique(data.outer_folds))
    }
    return {
        "metrics": compute_patient_metrics(data.targets, data.probabilities, CLINICAL_CLASSES),
        "macro_f1_95_ci": list(
            bootstrap_macro_f1_ci(
                data.targets, data.probabilities, n_bootstrap=samples, seed=seed
            )
        ),
        "fold_metrics": fold_metrics,
    }


def evaluate_d0_oof(
    config: Mapping[str, Any],
    candidate: D0OOFData,
    baseline: D0OOFData,
    attention_audit: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
) -> dict[str, Any]:
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    candidate_report = _model_report(candidate, samples, seed)
    baseline_report = _model_report(baseline, samples, seed)
    frozen_baseline = config["evaluation"]["baseline"]
    for name, config_key in (
        ("macro_f1", "macro_f1"),
        ("ece", "ece"),
        ("multiclass_brier", "multiclass_brier"),
    ):
        if not np.isclose(
            baseline_report["metrics"][name],
            float(frozen_baseline[config_key]),
            atol=5e-10,
        ):
            raise ValueError(f"D0 recomputed baseline {name} differs from preregistration")
    paired = paired_bootstrap_macro_f1(
        candidate.targets,
        baseline.probabilities,
        candidate.probabilities,
        n_bootstrap=samples,
        seed=seed,
    )
    fold_deltas = {
        fold: float(
            candidate_report["fold_metrics"][fold]["macro_f1"]
            - baseline_report["fold_metrics"][fold]["macro_f1"]
        )
        for fold in candidate_report["fold_metrics"]
    }
    class_deltas = {
        candidate_row["class_name"]: float(candidate_row["f1"] - baseline_row["f1"])
        for candidate_row, baseline_row in zip(
            candidate_report["metrics"]["per_class"],
            baseline_report["metrics"]["per_class"],
        )
    }
    gates = config["gates"]
    metrics = candidate_report["metrics"]
    checks = {
        "minimum_macro_f1": metrics["macro_f1"] >= float(gates["minimum_macro_f1"]),
        "minimum_delta_vs_e2_conditional": paired["observed_delta"]
        >= float(gates["minimum_macro_f1_delta_vs_e2_conditional"]),
        "paired_ci_lower_above_zero": paired["ci_low"] > 0,
        "minimum_positive_folds": sum(value > 0 for value in fold_deltas.values())
        >= int(gates["minimum_positive_folds"]),
        "per_class_f1_safety": min(class_deltas.values())
        >= -float(gates["maximum_per_class_f1_drop"]),
        "minimum_each_class_f1": min(
            row["f1"] for row in metrics["per_class"]
        )
        >= float(gates["minimum_each_class_f1"]),
        "maximum_ece": metrics["ece"] <= float(gates["maximum_ece"]),
        "maximum_multiclass_brier": metrics["multiclass_brier"]
        <= float(gates["maximum_multiclass_brier"]),
        "attention_collapse": attention_audit["pooled"]["multi_image_collapse_rate"]
        <= float(gates["maximum_multi_image_attention_collapse_rate"]),
        "resource_and_isolation_contract": bool(resource_audit["all_passed"]),
    }
    all_passed = all(checks.values())
    return {
        "contract": {
            "prediction_level": "patient",
            "patients": len(candidate.person_keys),
            "classes": list(CLINICAL_CLASSES),
            "outer_folds": sorted(int(value) for value in np.unique(candidate.outer_folds)),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        },
        "models": {"D0": candidate_report, "E2_conditional": baseline_report},
        "d0_minus_e2_conditional": {
            **paired,
            "fold_macro_f1_deltas": fold_deltas,
            "positive_folds": sum(value > 0 for value in fold_deltas.values()),
            "per_class_f1_deltas": class_deltas,
        },
        "attention_audit": attention_audit,
        "resource_audit": dict(resource_audit),
        "gate": {
            "checks": checks,
            "all_passed": all_passed,
            "decision": "advance_to_f0_f1" if all_passed else "stop_without_fusion_training",
        },
    }


def audit_d0_resources(
    config: Mapping[str, Any],
    project_root: Path,
    experiment_dir: Path,
    pilot_path: Path,
    summary_paths: Sequence[Path],
    fold_paths: Sequence[Path],
    attention_paths: Sequence[Path],
) -> dict[str, Any]:
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    expected_folds = list(config["evaluation"]["outer_folds"])
    if [int(summary["outer_fold"]) for summary in summaries] != expected_folds:
        raise ValueError("D0 formal summaries are not in frozen fold order")
    if not (len(summary_paths) == len(fold_paths) == len(attention_paths)):
        raise ValueError("D0 formal artifact lists have different lengths")

    pilot_history_path = experiment_dir / "logs" / "D0" / "fold_0_pilot_history.json"
    pilot_history = json.loads(pilot_history_path.read_text(encoding="utf-8"))
    pilot_cpu_samples = [float(row["resource"]["cpu_average_percent"]) for row in pilot_history]

    artifact_checks: list[dict[str, Any]] = []
    expected_config_sha = sha256_file(project_root / str(summaries[0]["config_path"]))
    expected_pretrained_sha = str(config["model"]["pretrained_sha256"])
    for path, attention_path, summary in zip(fold_paths, attention_paths, summaries):
        checkpoint_path = project_root / str(summary["best_checkpoint_path"])
        checks = {
            "identity": summary["experiment_code"] == "D0"
            and summary["data_fingerprint"] == str(config["data_fingerprint"])
            and int(summary["seed"])
            == int(config["evaluation"]["seeds"][int(summary["outer_fold"])]),
            "config": summary["config_sha256"] == expected_config_sha,
            "pretrained": summary["pretrained_sha256"] == expected_pretrained_sha,
            "prediction": summary["prediction_sha256"] == sha256_file(path),
            "attention": summary["attention_sha256"] == sha256_file(attention_path),
            "checkpoint": checkpoint_path.is_file()
            and summary["best_checkpoint_sha256"] == sha256_file(checkpoint_path),
            "mlflow": bool(summary["mlflow_parent_run_id"])
            and bool(summary["mlflow_fold_run_id"]),
        }
        artifact_checks.append(
            {"fold": int(summary["outer_fold"]), "checks": checks, "all_passed": all(checks.values())}
        )
    pilot_checks = {
        "completed_five_epochs": pilot["status"] == "COMPLETED"
        and int(pilot["epochs_completed"]) == int(config["training"]["pilot_epochs"]),
        "under_one_hour": float(pilot["elapsed_hours_total"])
        < float(config["stopping"]["pilot_requires_under_hours"]),
        "outer_test_not_iterated": pilot["outer_test_iterated"] is False,
        "gpu_memory": float(pilot["peak_gpu_memory_reserved_gb"])
        <= float(config["runtime"]["max_gpu_memory_gb"]),
        "cpu_not_sustained_saturation": bool(pilot_cpu_samples)
        and max(pilot_cpu_samples) < 95.0,
        "clean_worktree": pilot["git_dirty"] is False,
        "config_and_weights": pilot["config_sha256"] == expected_config_sha
        and pilot["pretrained_sha256"] == expected_pretrained_sha,
    }
    formal_checks = {
        "all_folds_completed": all(
            summary["status"] in {"COMPLETED", "EARLY_STOPPED"} for summary in summaries
        ),
        "outer_test_isolation": all(
            summary["outer_test_used_for_training_or_early_stopping"] is False
            for summary in summaries
        ),
        "clean_training_worktrees": all(summary["git_dirty"] is False for summary in summaries),
        "gpu_memory": max(float(summary["peak_gpu_memory_reserved_gb"]) for summary in summaries)
        <= float(config["runtime"]["max_gpu_memory_gb"]),
        "runtime": sum(float(summary["elapsed_hours_total"]) for summary in summaries)
        <= float(config["runtime"]["soft_limit_hours"]),
        "artifact_hashes_and_mlflow": all(item["all_passed"] for item in artifact_checks),
        "tracking_database": (experiment_dir / "tracking" / "mlflow.db").is_file(),
    }
    return {
        "pilot_path": pilot_path.as_posix(),
        "pilot_sha256": sha256_file(pilot_path),
        "pilot_history_path": pilot_history_path.as_posix(),
        "pilot_history_sha256": sha256_file(pilot_history_path),
        "pilot_cpu_samples_percent": pilot_cpu_samples,
        "formal_summaries": [
            {"fold": int(summary["outer_fold"]), "path": path.as_posix(), "sha256": sha256_file(path)}
            for path, summary in zip(summary_paths, summaries)
        ],
        "pilot_checks": pilot_checks,
        "formal_checks": formal_checks,
        "formal_artifact_checks": artifact_checks,
        "formal_elapsed_hours_total": sum(
            float(summary["elapsed_hours_total"]) for summary in summaries
        ),
        "formal_peak_gpu_memory_reserved_gb": max(
            float(summary["peak_gpu_memory_reserved_gb"]) for summary in summaries
        ),
        "all_passed": all(pilot_checks.values()) and all(formal_checks.values()),
    }


def write_merged_oof(data: D0OOFData, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"D0 merged OOF already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data.rows[0]))
            writer.writeheader()
            writer.writerows(data.rows)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def write_result(report: Mapping[str, Any], output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"D0 evaluation already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def git_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return revision, dirty


def run_d0_evaluation(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    experiment_dir: Path,
) -> dict[str, Any]:
    folds = list(config["evaluation"]["outer_folds"])
    seeds = list(config["evaluation"]["seeds"])
    fold_paths = [experiment_dir / "reports" / "oof" / f"D0_fold{fold}.csv" for fold in folds]
    attention_paths = [
        experiment_dir / "reports" / "attention" / f"D0_fold{fold}.csv" for fold in folds
    ]
    summary_paths = [
        experiment_dir / "reports" / f"D0-fold{fold}-seed{seed}-formal_summary.json"
        for fold, seed in zip(folds, seeds)
    ]
    pilot_path = experiment_dir / "reports" / "D0-fold0-seed20260724-pilot_summary.json"
    baseline_path = (project_root / str(config["evaluation"]["baseline"]["path"])).resolve()
    candidate, baseline, oof_inputs = load_d0_oof_inputs(fold_paths, baseline_path, config)
    attention_rows, attention_inputs = read_attention_files(attention_paths)
    attention = {
        "inputs": attention_inputs,
        **audit_attention_rows(
            attention_rows,
            collapse_threshold=float(config["model"]["attention_collapse_threshold"]),
            max_collapse_rate=float(config["gates"]["maximum_multi_image_attention_collapse_rate"]),
        ),
    }
    resources = audit_d0_resources(
        config,
        project_root,
        experiment_dir,
        pilot_path,
        summary_paths,
        fold_paths,
        attention_paths,
    )
    result = evaluate_d0_oof(config, candidate, baseline, attention, resources)
    merged_path = experiment_dir / "reports" / "oof" / "D0_oof.csv"
    result_path = experiment_dir / "reports" / "D0-oof-evaluation.json"
    revision, dirty = git_state(project_root)
    report = {
        "experiment_code": "D0",
        "status": "COMPLETED",
        "data_fingerprint": str(config["data_fingerprint"]),
        "git_revision": revision,
        "git_dirty_at_evaluation": dirty,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "oof_inputs": oof_inputs,
        "baseline_path": baseline_path.relative_to(project_root).as_posix(),
        "baseline_sha256": sha256_file(baseline_path),
        "result": result,
    }
    write_merged_oof(candidate, merged_path)
    report["merged_oof_path"] = merged_path.relative_to(project_root).as_posix()
    report["merged_oof_sha256"] = sha256_file(merged_path)
    write_result(report, result_path)
    return {
        "report": report,
        "merged_oof_path": merged_path,
        "result_path": result_path,
        "result_sha256": sha256_file(result_path),
    }
