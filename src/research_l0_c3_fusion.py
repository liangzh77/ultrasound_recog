"""Frozen L0/C3 abnormal-condition OOF fusion sensitivity analysis."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.research_clinical import CLINICAL_CLASSES
from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)


CLASS_SLUGS = ("ra", "ga", "spa", "oa", "injury")


@dataclass(frozen=True)
class L1Inputs:
    person_keys: tuple[str, ...]
    outer_folds: np.ndarray
    targets: np.ndarray
    reference_classes: tuple[str, ...]
    l0_probabilities: np.ndarray
    c3_probabilities: np.ndarray
    fused_probabilities: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("study_code") != "L1":
        raise ValueError("L1 config has an invalid study code")
    if config.get("status") != "frozen_after_l0_before_l1_results":
        raise ValueError("L1 method is not frozen")
    if tuple(config.get("classes", ())) != CLASS_SLUGS:
        raise ValueError("L1 class order differs from the frozen contract")
    fusion = config.get("fixed_fusion", {})
    clinical_weight = float(fusion.get("clinical_weight", -1))
    image_weight = float(fusion.get("image_weight", -1))
    if fusion.get("search_weights") is not False:
        raise ValueError("L1 forbids fusion-weight search")
    if clinical_weight < 0 or image_weight < 0 or not np.isclose(
        clinical_weight + image_weight, 1.0
    ):
        raise ValueError("L1 weights must be nonnegative and sum to one")
    interpretation = config.get("interpretation_contract", {})
    if interpretation.get("allow_2026") is not False:
        raise ValueError("L1 must not access the 2026 cohort")
    return config


def _resolve(root: Path, value: str) -> Path:
    project_root = root.resolve()
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("L1 path escapes the project root") from error
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("L1 input OOF is empty")
    return rows


def _index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = row.get("person_key", "").strip()
        if not key or key in result:
            raise ValueError(f"L1 {label} OOF has empty or duplicate person keys")
        result[key] = row
    return result


def _validate_probabilities(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("L1 probabilities are invalid")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=tolerance):
        raise ValueError("L1 probabilities do not sum to one")
    return matrix


def load_inputs(config_path: Path, project_root: Path) -> tuple[dict[str, Any], L1Inputs]:
    config = load_config(config_path)
    image_spec: Mapping[str, Any] = config["inputs"]["image_oof"]
    clinical_spec: Mapping[str, Any] = config["inputs"]["clinical_oof"]
    image_path = _resolve(project_root, str(image_spec["path"]))
    clinical_path = _resolve(project_root, str(clinical_spec["path"]))
    for path, spec in ((image_path, image_spec), (clinical_path, clinical_spec)):
        if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
            raise ValueError("L1 input SHA-256 contract failed")

    image_rows = [row for row in _read_rows(image_path) if int(row["reference_id"]) != 0]
    clinical_rows = _read_rows(clinical_path)
    image_index = _index(image_rows, "L0")
    clinical_index = _index(clinical_rows, "C3")
    expected = int(config["cohort_contract"]["expected_patients"])
    if len(image_index) != expected or len(clinical_index) != expected:
        raise ValueError("L1 OOF patient count differs from the frozen contract")
    if set(image_index) != set(clinical_index):
        raise ValueError("L1 OOF patient sets differ")

    person_keys = tuple(sorted(image_index))
    folds: list[int] = []
    targets: list[int] = []
    references: list[str] = []
    raw_l0: list[list[float]] = []
    raw_c3: list[list[float]] = []
    image_columns = tuple(image_spec["probability_columns"])
    clinical_columns = tuple(clinical_spec["probability_columns"])
    for key in person_keys:
        image = image_index[key]
        clinical = clinical_index[key]
        image_fold = int(image["outer_fold"])
        clinical_fold = int(clinical["outer_fold"])
        image_target = int(image["reference_id"])
        clinical_target = int(clinical["reference_id"])
        if (
            image_fold != clinical_fold
            or image_target != clinical_target + 1
            or image["reference_class"] != clinical["reference_class"]
        ):
            raise ValueError("L1 paired fold or reference contract failed")
        folds.append(image_fold)
        targets.append(clinical_target)
        references.append(clinical["reference_class"])
        raw_l0.append([float(image[column]) for column in image_columns])
        raw_c3.append([float(clinical[column]) for column in clinical_columns])

    if set(folds) != set(config["cohort_contract"]["expected_outer_folds"]):
        raise ValueError("L1 outer folds differ from the frozen contract")
    tolerance = float(config["probability_contract"]["tolerance"])
    l0 = np.asarray(raw_l0, dtype=np.float64)
    denominator = l0.sum(axis=1, keepdims=True)
    if (denominator <= float(config["probability_contract"]["minimum_denominator"])).any():
        raise ValueError("L1 conditional L0 denominator is invalid")
    l0 = _validate_probabilities(l0 / denominator, tolerance)
    c3 = _validate_probabilities(np.asarray(raw_c3, dtype=np.float64), tolerance)
    fusion = config["fixed_fusion"]
    fused = _validate_probabilities(
        float(fusion["clinical_weight"]) * c3
        + float(fusion["image_weight"]) * l0,
        tolerance,
    )
    return config, L1Inputs(
        person_keys=person_keys,
        outer_folds=np.asarray(folds, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        reference_classes=tuple(references),
        l0_probabilities=l0,
        c3_probabilities=c3,
        fused_probabilities=fused,
    )


def _model_report(data: L1Inputs, probabilities: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    return {
        "metrics": compute_patient_metrics(data.targets, probabilities, CLINICAL_CLASSES),
        "macro_f1_95_ci": list(
            bootstrap_macro_f1_ci(
                data.targets, probabilities, n_bootstrap=samples, seed=seed
            )
        ),
        "fold_metrics": {
            str(fold): compute_patient_metrics(
                data.targets[data.outer_folds == fold],
                probabilities[data.outer_folds == fold],
                CLINICAL_CLASSES,
            )
            for fold in sorted(np.unique(data.outer_folds))
        },
    }


def evaluate(config: Mapping[str, Any], data: L1Inputs) -> dict[str, Any]:
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    reports = {
        "L0_conditional": _model_report(data, data.l0_probabilities, samples, seed),
        "C3": _model_report(data, data.c3_probabilities, samples, seed),
        "L1_fixed_fusion": _model_report(data, data.fused_probabilities, samples, seed),
    }
    paired = paired_bootstrap_macro_f1(
        data.targets,
        data.c3_probabilities,
        data.fused_probabilities,
        n_bootstrap=samples,
        seed=seed,
    )
    fold_deltas = {
        fold: reports["L1_fixed_fusion"]["fold_metrics"][fold]["macro_f1"]
        - reports["C3"]["fold_metrics"][fold]["macro_f1"]
        for fold in reports["C3"]["fold_metrics"]
    }
    per_class_deltas = {
        baseline["class_name"]: float(candidate["f1"] - baseline["f1"])
        for baseline, candidate in zip(
            reports["C3"]["metrics"]["per_class"],
            reports["L1_fixed_fusion"]["metrics"]["per_class"],
            strict=True,
        )
    }
    gate = config["decision_gate"]
    checks = {
        "minimum_macro_f1_delta": paired["observed_delta"]
        >= float(gate["minimum_macro_f1_delta_vs_c3"]),
        "paired_ci_lower_above_zero": paired["ci_low"] > 0,
        "minimum_positive_folds": sum(value > 0 for value in fold_deltas.values())
        >= int(gate["minimum_positive_folds"]),
        "per_class_safety": min(per_class_deltas.values())
        >= -float(gate["maximum_per_class_f1_drop"]),
    }
    return {
        "patients": len(data.person_keys),
        "models": reports,
        "fixed_fusion_minus_c3": {
            **paired,
            "fold_macro_f1_deltas": fold_deltas,
            "per_class_f1_deltas": per_class_deltas,
        },
        "decision_gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "decision": "credible_descriptive_complementarity" if all(checks.values()) else "stop_l0_c3_route",
        },
    }


def patient_rows(data: L1Inputs) -> list[dict[str, Any]]:
    rows = []
    for index, key in enumerate(data.person_keys):
        row: dict[str, Any] = {
            "prediction_level": "patient_abnormal_sensitivity",
            "person_key": key,
            "outer_fold": int(data.outer_folds[index]),
            "reference_class": data.reference_classes[index],
            "reference_id": int(data.targets[index]),
        }
        for prefix, probabilities in (
            ("l0_conditional", data.l0_probabilities[index]),
            ("c3", data.c3_probabilities[index]),
            ("l1_fixed_fusion", data.fused_probabilities[index]),
        ):
            for slug, value in zip(CLASS_SLUGS, probabilities, strict=True):
                row[f"{prefix}_prob_{slug}"] = float(value)
        rows.append(row)
    return rows
