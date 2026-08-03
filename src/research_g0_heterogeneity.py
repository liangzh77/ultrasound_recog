"""Fold-safe, privacy-bounded audit helpers for the fixed G0 OOF result."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.research_proxy_audit import FEATURE_GROUPS, ProxyTable, aggregate_patient_proxy_features


PROBABILITY_COLUMNS = (
    "prob_normal",
    "prob_ra",
    "prob_ga",
    "prob_spa",
    "prob_oa",
    "prob_injury",
)


@dataclass(frozen=True)
class H0Inputs:
    """Validated, pseudonymous patient inputs; no source paths or free text."""

    g0_rows: tuple[dict[str, str], ...]
    proxy_tables: Mapping[str, ProxyTable]
    proxy_oof_rows: Mapping[str, tuple[dict[str, str], ...]]
    input_hashes: Mapping[str, str]
    images: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_input_path(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("H0 input paths must be project-relative")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("H0 input path escapes the project root") from error
    if not path.is_file():
        raise FileNotFoundError("A frozen H0 input is missing")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_file(
    root: Path,
    item: Mapping[str, Any],
    label: str,
) -> tuple[Path, str]:
    path = _safe_input_path(root, str(item["path"]))
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise ValueError(f"Frozen H0 input hash mismatch: {label}")
    return path, actual


def _validate_g0_rows(
    rows: list[dict[str, str]], expected: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    allowed = {
        "person_key",
        "outer_fold",
        "reference_id",
        "predicted_id",
        "prob_abnormal",
        "image_count",
    }
    missing = allowed.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"G0 OOF is missing required columns: {sorted(missing)}")
    if len(rows) != int(expected["patients"]):
        raise ValueError("G0 OOF patient count changed")
    by_person = {row["person_key"]: row for row in rows}
    if len(by_person) != len(rows):
        raise ValueError("G0 OOF patient keys are not unique")
    folds = {int(row["outer_fold"]) for row in rows}
    if folds != set(map(int, expected["folds"])):
        raise ValueError("G0 OOF fold set changed")
    references = np.asarray([int(row["reference_id"]) for row in rows])
    if set(references) != {0, 1}:
        raise ValueError("G0 OOF reference must be binary")
    if int(np.sum(references == 0)) != int(expected["normal_patients"]):
        raise ValueError("G0 OOF normal count changed")
    if int(np.sum(references == 1)) != int(expected["abnormal_patients"]):
        raise ValueError("G0 OOF abnormal count changed")
    probabilities = np.asarray([float(row["prob_abnormal"]) for row in rows])
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("G0 OOF probabilities are invalid")
    return by_person


def load_h0_inputs(root: Path, config: Mapping[str, Any]) -> H0Inputs:
    """Validate frozen hashes and connect only pseudonymous numerical inputs."""
    inputs = config["inputs"]
    hashes: dict[str, str] = {}
    g0_path, hashes["g0_oof"] = _verify_file(root, inputs["g0_oof"], "g0_oof")
    _, hashes["g0_evaluation"] = _verify_file(
        root, inputs["g0_evaluation"], "g0_evaluation"
    )
    feature_path, hashes["image_proxy_features"] = _verify_file(
        root, inputs["image_proxy_features"], "image_proxy_features"
    )
    _, hashes["model_visible_proxy_audit"] = _verify_file(
        root, inputs["model_visible_proxy_audit"], "model_visible_proxy_audit"
    )

    g0_rows = _read_csv(g0_path)
    g0_by_person = _validate_g0_rows(g0_rows, config["expected"])
    image_rows = _read_csv(feature_path)
    if len(image_rows) != int(config["expected"]["images"]):
        raise ValueError("Proxy image count changed")
    if len({row["image_key"] for row in image_rows}) != len(image_rows):
        raise ValueError("Proxy image keys are not unique")

    proxy_tables: dict[str, ProxyTable] = {}
    proxy_oofs: dict[str, tuple[dict[str, str], ...]] = {}
    for group, group_config in config["proxy_groups"].items():
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown frozen proxy group: {group}")
        table = aggregate_patient_proxy_features(image_rows, FEATURE_GROUPS[group])
        if set(table.person_keys) != set(g0_by_person):
            raise ValueError(f"Proxy patient coverage changed: {group}")
        for index, person_key in enumerate(table.person_keys):
            g0_row = g0_by_person[person_key]
            if int(table.outer_folds[index]) != int(g0_row["outer_fold"]):
                raise ValueError(f"Proxy fold mismatch: {group}")
            expected_binary = 0 if int(table.targets[index]) == 0 else 1
            if expected_binary != int(g0_row["reference_id"]):
                raise ValueError(f"Proxy reference mismatch: {group}")
        oof_item = {
            "path": group_config["oof_path"],
            "sha256": group_config["oof_sha256"],
        }
        oof_path, hashes[f"proxy_oof:{group}"] = _verify_file(
            root, oof_item, f"proxy_oof:{group}"
        )
        oof_rows = _read_csv(oof_path)
        if len(oof_rows) != len(g0_rows):
            raise ValueError(f"Proxy OOF patient count changed: {group}")
        oof_by_person = {row["person_key"]: row for row in oof_rows}
        if set(oof_by_person) != set(g0_by_person):
            raise ValueError(f"Proxy OOF coverage changed: {group}")
        for person_key, row in oof_by_person.items():
            g0_row = g0_by_person[person_key]
            probabilities = np.asarray([float(row[name]) for name in PROBABILITY_COLUMNS])
            if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Proxy OOF probabilities invalid: {group}")
            if int(row["outer_fold"]) != int(g0_row["outer_fold"]):
                raise ValueError(f"Proxy OOF fold mismatch: {group}")
            expected_binary = 0 if int(row["reference_id"]) == 0 else 1
            if expected_binary != int(g0_row["reference_id"]):
                raise ValueError(f"Proxy OOF reference mismatch: {group}")
        proxy_tables[group] = table
        proxy_oofs[group] = tuple(oof_by_person[key] for key in sorted(oof_by_person))

    return H0Inputs(
        g0_rows=tuple(g0_by_person[key] for key in sorted(g0_by_person)),
        proxy_tables=proxy_tables,
        proxy_oof_rows=proxy_oofs,
        input_hashes=hashes,
        images=len(image_rows),
    )


def safe_input_summary(inputs: H0Inputs) -> dict[str, Any]:
    return {
        "status": "READY",
        "patients": len(inputs.g0_rows),
        "images": inputs.images,
        "folds": sorted({int(row["outer_fold"]) for row in inputs.g0_rows}),
        "proxy_groups": sorted(inputs.proxy_tables),
        "input_hashes": dict(sorted(inputs.input_hashes.items())),
        "privacy_boundary": "pseudonymous_numerical_inputs_only",
    }
