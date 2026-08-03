"""Strict contracts for abnormal-patient image/clinical OOF fusion research."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.research_clinical import CLINICAL_CLASSES


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
