"""Strict contracts for abnormal-patient image/clinical OOF fusion research."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil
import yaml

from src.research_clinical import CLINICAL_CLASSES
from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)
from src.research_tracking import LocalResearchTracker


CLASS_SLUGS = ("ra", "ga", "spa", "oa", "injury")


@dataclass(frozen=True)
class X0Inputs:
    person_keys: tuple[str, ...]
    outer_folds: np.ndarray
    targets: np.ndarray
    reference_classes: tuple[str, ...]
    image_probabilities: np.ndarray
    clinical_probabilities: np.ndarray
    fused_probabilities: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_x0_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("study_code") != "X0":
        raise ValueError("X0 config has an invalid study code")
    if tuple(payload.get("classes", ())) != CLASS_SLUGS:
        raise ValueError("X0 class order differs from the frozen contract")
    fusion = payload.get("primary_fixed_fusion", {})
    clinical_weight = float(fusion.get("clinical_weight", -1))
    image_weight = float(fusion.get("image_weight", -1))
    if fusion.get("search_weights") is not False:
        raise ValueError("X0 forbids fusion-weight search")
    if clinical_weight < 0 or image_weight < 0 or not np.isclose(
        clinical_weight + image_weight, 1.0
    ):
        raise ValueError("X0 fusion weights must be nonnegative and sum to one")
    return payload


def _resolve_project_path(project_root: Path, value: str) -> Path:
    root = project_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("X0 input path escapes the project root") from error
    return candidate


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("X0 input OOF is empty")
    return rows


def _index_unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("person_key", "").strip()
        if not key:
            raise ValueError(f"{label} OOF has an empty person key")
        if key in result:
            raise ValueError(f"{label} OOF has duplicate person keys")
        result[key] = row
    return result


def _validate_probability_matrix(
    probabilities: np.ndarray, tolerance: float
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("X0 probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("X0 probabilities must be between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=tolerance):
        raise ValueError("X0 probabilities do not sum to one")
    return probabilities


def _probability_matrix(
    rows: list[dict[str, str]], columns: tuple[str, ...], tolerance: float
) -> np.ndarray:
    try:
        probabilities = np.asarray(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("X0 OOF probability columns are invalid") from error
    return _validate_probability_matrix(probabilities, tolerance)


def load_x0_inputs(
    config_path: Path, project_root: Path
) -> tuple[dict[str, Any], X0Inputs]:
    config = load_x0_config(config_path)
    tolerance = float(config["probability_contract"]["tolerance"])
    image_spec: Mapping[str, Any] = config["inputs"]["image_oof"]
    clinical_spec: Mapping[str, Any] = config["inputs"]["clinical_oof"]
    image_path = _resolve_project_path(project_root, str(image_spec["path"]))
    clinical_path = _resolve_project_path(project_root, str(clinical_spec["path"]))
    for path, spec in ((image_path, image_spec), (clinical_path, clinical_spec)):
        if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
            raise ValueError("X0 input SHA-256 contract failed")

    image_all = _read_rows(image_path)
    image_rows = [row for row in image_all if int(row["reference_id"]) != 0]
    clinical_rows = _read_rows(clinical_path)
    image_index = _index_unique(image_rows, "image")
    clinical_index = _index_unique(clinical_rows, "clinical")
    expected = int(config["cohort_contract"]["expected_patients"])
    if len(image_index) != expected or len(clinical_index) != expected:
        raise ValueError("X0 OOF patient count differs from the frozen contract")
    if set(image_index) != set(clinical_index):
        raise ValueError("X0 image and clinical OOF cover different patients")

    keys = tuple(sorted(image_index))
    ordered_image = [image_index[key] for key in keys]
    ordered_clinical = [clinical_index[key] for key in keys]
    folds = []
    targets = []
    references = []
    for image_row, clinical_row in zip(ordered_image, ordered_clinical):
        image_fold = int(image_row["outer_fold"])
        clinical_fold = int(clinical_row["outer_fold"])
        image_target = int(image_row["reference_id"])
        clinical_target = int(clinical_row["reference_id"])
        if (
            image_fold != clinical_fold
            or image_target != clinical_target + 1
            or image_row["reference_class"] != clinical_row["reference_class"]
        ):
            raise ValueError("X0 paired OOF fold or reference contract failed")
        if clinical_target not in range(len(CLINICAL_CLASSES)):
            raise ValueError("X0 clinical reference ID is outside the class contract")
        folds.append(image_fold)
        targets.append(clinical_target)
        references.append(clinical_row["reference_class"])

    expected_folds = set(int(value) for value in config["cohort_contract"]["expected_outer_folds"])
    if set(folds) != expected_folds:
        raise ValueError("X0 outer folds differ from the frozen contract")

    image_columns = tuple(str(value) for value in image_spec["probability_columns"])
    clinical_columns = tuple(str(value) for value in clinical_spec["probability_columns"])
    raw_image = np.asarray(
        [[float(row[column]) for column in image_columns] for row in ordered_image],
        dtype=np.float64,
    )
    minimum = float(config["probability_contract"]["minimum_denominator"])
    denominator = raw_image.sum(axis=1, keepdims=True)
    if not np.isfinite(raw_image).all() or (raw_image < 0).any() or (denominator <= minimum).any():
        raise ValueError("X0 conditional image probability denominator is invalid")
    image_probabilities = _validate_probability_matrix(raw_image / denominator, tolerance)
    clinical_probabilities = _probability_matrix(
        ordered_clinical, clinical_columns, tolerance
    )
    fusion = config["primary_fixed_fusion"]
    fused = (
        float(fusion["clinical_weight"]) * clinical_probabilities
        + float(fusion["image_weight"]) * image_probabilities
    )
    _validate_probability_matrix(fused, tolerance)
    return config, X0Inputs(
        person_keys=keys,
        outer_folds=np.asarray(folds, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        reference_classes=tuple(references),
        image_probabilities=image_probabilities,
        clinical_probabilities=clinical_probabilities,
        fused_probabilities=fused,
    )


def _model_report(
    data: X0Inputs,
    probabilities: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    interval = bootstrap_macro_f1_ci(
        data.targets,
        probabilities,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "metrics": compute_patient_metrics(
            data.targets, probabilities, CLINICAL_CLASSES
        ),
        "macro_f1_95_ci": list(interval),
        "fold_metrics": {
            str(fold): compute_patient_metrics(
                data.targets[data.outer_folds == fold],
                probabilities[data.outer_folds == fold],
                CLINICAL_CLASSES,
            )
            for fold in sorted(np.unique(data.outer_folds))
        },
    }


def _error_complementarity(data: X0Inputs) -> dict[str, Any]:
    targets = data.targets
    image_predictions = data.image_probabilities.argmax(axis=1)
    clinical_predictions = data.clinical_probabilities.argmax(axis=1)
    clinical_wrong = clinical_predictions != targets
    image_wrong = image_predictions != targets
    rescued = clinical_wrong & (image_predictions == targets)
    reverse_rescued = image_wrong & (clinical_predictions == targets)
    clinical_error_count = int(clinical_wrong.sum())
    image_error_count = int(image_wrong.sum())
    return {
        "clinical_error_count": clinical_error_count,
        "clinical_errors_rescued_by_image": int(rescued.sum()),
        "clinical_error_rescue_fraction_by_image": (
            float(rescued.sum() / clinical_error_count) if clinical_error_count else 0.0
        ),
        "folds_with_clinical_error_rescue": sorted(
            int(value) for value in np.unique(data.outer_folds[rescued])
        ),
        "classes_with_clinical_error_rescue": [
            CLINICAL_CLASSES[int(value)] for value in sorted(np.unique(targets[rescued]))
        ],
        "image_error_count": image_error_count,
        "image_errors_rescued_by_clinical": int(reverse_rescued.sum()),
        "image_error_rescue_fraction_by_clinical": (
            float(reverse_rescued.sum() / image_error_count) if image_error_count else 0.0
        ),
        "both_wrong_count": int((clinical_wrong & image_wrong).sum()),
        "both_correct_count": int(((~clinical_wrong) & (~image_wrong)).sum()),
    }


def evaluate_x0(config: Mapping[str, Any], data: X0Inputs) -> dict[str, Any]:
    evaluation = config["evaluation"]
    samples = int(evaluation["bootstrap_samples"])
    seed = int(evaluation["bootstrap_seed"])
    reports = {
        "E2_conditional": _model_report(
            data, data.image_probabilities, samples, seed
        ),
        "C3": _model_report(data, data.clinical_probabilities, samples, seed),
        "X0_fixed_fusion": _model_report(
            data, data.fused_probabilities, samples, seed
        ),
    }
    paired = paired_bootstrap_macro_f1(
        data.targets,
        data.clinical_probabilities,
        data.fused_probabilities,
        n_bootstrap=samples,
        seed=seed,
    )
    fold_deltas = {
        fold: (
            reports["X0_fixed_fusion"]["fold_metrics"][fold]["macro_f1"]
            - reports["C3"]["fold_metrics"][fold]["macro_f1"]
        )
        for fold in reports["C3"]["fold_metrics"]
    }
    c3_per_class = reports["C3"]["metrics"]["per_class"]
    fusion_per_class = reports["X0_fixed_fusion"]["metrics"]["per_class"]
    per_class_f1_deltas = {
        baseline["class_name"]: float(candidate["f1"] - baseline["f1"])
        for baseline, candidate in zip(c3_per_class, fusion_per_class)
    }
    complementarity = _error_complementarity(data)

    gate = config["d0_feasibility_gate"]
    blend_gate = gate["primary_blend"]
    blend_checks = {
        "macro_f1_delta": paired["observed_delta"]
        >= float(blend_gate["minimum_macro_f1_delta_vs_c3"]),
        "paired_ci_lower_above_zero": paired["ci_low"] > 0,
        "positive_folds": sum(delta > 0 for delta in fold_deltas.values())
        >= int(blend_gate["minimum_positive_folds"]),
        "per_class_safety": min(per_class_f1_deltas.values())
        >= -float(blend_gate["maximum_per_class_f1_drop"]),
    }
    rescue_gate = gate["error_rescue"]
    rescue_checks = {
        "rescue_fraction": complementarity["clinical_error_rescue_fraction_by_image"]
        >= float(rescue_gate["minimum_c3_error_rescue_fraction_by_e2"]),
        "fold_coverage": len(complementarity["folds_with_clinical_error_rescue"])
        >= int(rescue_gate["minimum_folds_with_at_least_one_rescue"]),
        "class_coverage": len(complementarity["classes_with_clinical_error_rescue"])
        >= int(rescue_gate["minimum_classes_with_at_least_one_rescue"]),
    }
    blend_passed = all(blend_checks.values())
    rescue_passed = all(rescue_checks.values())
    allow_d0 = blend_passed or rescue_passed
    return {
        "patients": len(data.person_keys),
        "outer_folds": sorted(int(value) for value in np.unique(data.outer_folds)),
        "models": reports,
        "fixed_fusion_minus_c3": {
            **paired,
            "fold_macro_f1_deltas": fold_deltas,
            "per_class_f1_deltas": per_class_f1_deltas,
        },
        "error_complementarity": complementarity,
        "d0_feasibility_gate": {
            "primary_blend_checks": blend_checks,
            "primary_blend_passed": blend_passed,
            "error_rescue_checks": rescue_checks,
            "error_rescue_passed": rescue_passed,
            "allow_d0_preregistration": allow_d0,
            "decision": (
                "advance_to_d0_preregistration" if allow_d0 else "stop_after_x0"
            ),
        },
    }


def _git_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _hardware_snapshot(max_cpu_threads: int) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "processor": platform.processor() or platform.machine(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": float(memory.total / (1024**3)),
        "memory_available_start_gb": float(memory.available / (1024**3)),
        "max_cpu_threads": max_cpu_threads,
        "gpu_used": False,
    }


def _patient_output_rows(data: X0Inputs) -> list[dict[str, Any]]:
    rows = []
    for index, key in enumerate(data.person_keys):
        target = int(data.targets[index])
        row: dict[str, Any] = {
            "prediction_level": "patient_abnormal_exploratory",
            "person_key": key,
            "outer_fold": int(data.outer_folds[index]),
            "reference_class": data.reference_classes[index],
            "reference_id": target,
            "e2_conditional_top1": int(data.image_probabilities[index].argmax()),
            "c3_top1": int(data.clinical_probabilities[index].argmax()),
            "x0_fixed_fusion_top1": int(data.fused_probabilities[index].argmax()),
        }
        for slug, value in zip(CLASS_SLUGS, data.image_probabilities[index]):
            row[f"e2_conditional_prob_{slug}"] = float(value)
        for slug, value in zip(CLASS_SLUGS, data.clinical_probabilities[index]):
            row[f"c3_prob_{slug}"] = float(value)
        for slug, value in zip(CLASS_SLUGS, data.fused_probabilities[index]):
            row[f"x0_fixed_fusion_prob_{slug}"] = float(value)
        row["e2_conditional_correct"] = int(row["e2_conditional_top1"] == target)
        row["c3_correct"] = int(row["c3_top1"] == target)
        row["x0_fixed_fusion_correct"] = int(row["x0_fixed_fusion_top1"] == target)
        rows.append(row)
    return rows


def _write_x0_artifacts(
    project_root: Path,
    config: Mapping[str, Any],
    data: X0Inputs,
    summary: dict[str, Any],
) -> tuple[Path, str, Path, str]:
    oof_path = _resolve_project_path(project_root, str(config["outputs"]["patient_oof_csv"]))
    result_path = _resolve_project_path(project_root, str(config["outputs"]["result_json"]))
    if oof_path.exists() or result_path.exists():
        raise FileExistsError("X0 formal outputs already exist and will not be overwritten")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    oof_temp = oof_path.with_suffix(oof_path.suffix + ".tmp")
    result_temp = result_path.with_suffix(result_path.suffix + ".tmp")
    rows = _patient_output_rows(data)
    try:
        with oof_temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        oof_sha = sha256_file(oof_temp)
        summary["patient_oof_path"] = oof_path.relative_to(project_root).as_posix()
        summary["patient_oof_sha256"] = oof_sha
        serialized = json.dumps(
            summary, ensure_ascii=False, indent=2, allow_nan=False
        )
        result_temp.write_text(serialized, encoding="utf-8")
        os.replace(oof_temp, oof_path)
        os.replace(result_temp, result_path)
    finally:
        oof_temp.unlink(missing_ok=True)
        result_temp.unlink(missing_ok=True)
    return oof_path, oof_sha, result_path, sha256_file(result_path)


def run_x0_formal(
    config_path: Path, project_root: Path, experiment_dir: Path
) -> dict[str, Any]:
    start = time.perf_counter()
    config, data = load_x0_inputs(config_path, project_root)
    revision, dirty = _git_state(project_root)
    if dirty:
        raise RuntimeError("Formal X0 requires a clean Git worktree")
    config_sha = sha256_file(config_path)
    runtime_config = config["runtime"]
    max_cpu_threads = int(runtime_config["max_cpu_threads"])
    hardware = _hardware_snapshot(max_cpu_threads)
    tracker = LocalResearchTracker(
        experiment_dir / "tracking", str(config["outputs"]["mlflow_experiment"])
    )
    metadata = {
        "study_code": "X0",
        "data_fingerprint": str(config["data_fingerprint"]),
        "config_sha256": config_sha,
        "image_oof_sha256": str(config["inputs"]["image_oof"]["sha256"]),
        "clinical_oof_sha256": str(config["inputs"]["clinical_oof"]["sha256"]),
        "git_revision": revision,
        "patients": len(data.person_keys),
        "max_cpu_threads": max_cpu_threads,
        "gpu_used": False,
    }
    with tracker.parent_run("X0-abnormal-oof-complementarity-formal", metadata) as run:
        report = evaluate_x0(config, data)
        elapsed = time.perf_counter() - start
        if elapsed > float(runtime_config["hard_limit_minutes"]) * 60:
            raise TimeoutError("X0 exceeded its frozen runtime limit")
        metrics = {
            "e2_conditional_macro_f1": report["models"]["E2_conditional"]["metrics"]["macro_f1"],
            "c3_macro_f1": report["models"]["C3"]["metrics"]["macro_f1"],
            "fixed_fusion_macro_f1": report["models"]["X0_fixed_fusion"]["metrics"]["macro_f1"],
            "fixed_fusion_delta_vs_c3": report["fixed_fusion_minus_c3"]["observed_delta"],
            "c3_error_rescue_fraction_by_e2": report["error_complementarity"]["clinical_error_rescue_fraction_by_image"],
            "allow_d0_preregistration": float(report["d0_feasibility_gate"]["allow_d0_preregistration"]),
            "runtime_seconds": elapsed,
        }
        tracker.log_metrics(metrics)
        summary = {
            "study_code": "X0",
            "status": "COMPLETED",
            "research_question": str(config["research_question"]),
            "data_fingerprint": str(config["data_fingerprint"]),
            "git_revision": revision,
            "git_dirty": dirty,
            "config_path": config_path.relative_to(project_root).as_posix(),
            "config_sha256": config_sha,
            "input_sha256": {
                "image_oof": str(config["inputs"]["image_oof"]["sha256"]),
                "clinical_oof": str(config["inputs"]["clinical_oof"]["sha256"]),
            },
            "seed": int(config["seed"]),
            "bootstrap_samples": int(config["evaluation"]["bootstrap_samples"]),
            "runtime_seconds": elapsed,
            "hardware": hardware,
            "mlflow_parent_run_id": run.info.run_id,
            "result": report,
        }
        oof_path, oof_sha, result_path, result_sha = _write_x0_artifacts(
            project_root, config, data, summary
        )
        tracker.client.set_tag(run.info.run_id, "patient_oof_sha256", oof_sha)
        tracker.client.set_tag(run.info.run_id, "result_json_sha256", result_sha)
        tracker.client.set_tag(
            run.info.run_id,
            "decision",
            report["d0_feasibility_gate"]["decision"],
        )
    return {
        "result": report,
        "patient_oof_path": oof_path,
        "patient_oof_sha256": oof_sha,
        "result_path": result_path,
        "result_sha256": result_sha,
        "mlflow_parent_run_id": run.info.run_id,
        "git_revision": revision,
        "config_sha256": config_sha,
    }
