"""Validation contract for continuous research documentation and provenance."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMON_REQUIRED = {
    "id",
    "kind",
    "status",
    "research_question",
    "preregistered_gate",
    "data_fingerprint",
    "git_commit",
    "command",
    "folds",
    "seeds",
    "pretrained_sha256",
    "hardware",
    "oof",
    "results",
    "mlflow_parent_run_ids",
    "markdown_report",
    "failure_or_negative_reason",
    "conclusion",
    "next_step",
}
FORMAL_MODEL_REQUIRED = {
    "config",
    "runtime_hours_by_fold",
    "peak_allocated_gpu_gb_by_fold",
    "fold_status",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "workspace/data/raw/",
    "workspace\\data\\raw\\",
    "/private/",
    "\\private\\",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(project_root: Path, revision: str, relative_path: str) -> str:
    data = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=project_root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(data).hexdigest()


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def _validate_privacy_paths(record: Mapping[str, Any]) -> None:
    for value in _all_strings(record):
        normalized = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
            raise ValueError(f"Ledger contains an absolute path: {value}")
        if any(fragment.replace("\\", "/") in normalized for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise ValueError("Ledger contains a raw/private patient path")


def _validate_hash(value: str, field: str) -> None:
    if not HEX_SHA256.fullmatch(str(value).casefold()):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _resolve_project_file(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Ledger path escapes project root") from error
    return candidate


def validate_experiment_record(
    record: Mapping[str, Any],
    project_root: Path,
    verify_artifacts: bool,
) -> None:
    missing = sorted(COMMON_REQUIRED - set(record))
    if record.get("kind") == "formal_model":
        missing.extend(sorted(FORMAL_MODEL_REQUIRED - set(record)))
    if missing:
        raise ValueError(f"Experiment {record.get('id')} missing fields: {', '.join(missing)}")
    _validate_privacy_paths(record)
    _validate_hash(str(record["data_fingerprint"]), "data_fingerprint")
    if not record["folds"] or not record["seeds"]:
        raise ValueError("folds and seeds must be non-empty")
    if record.get("kind") in {"formal_model", "resource_pilot"} and len(
        record["folds"]
    ) != len(record["seeds"]):
        raise ValueError("Model folds and seeds must have equal length")
    markdown_path = _resolve_project_file(project_root, str(record["markdown_report"]))
    if verify_artifacts and not markdown_path.is_file():
        raise ValueError(f"Missing Markdown report: {record['markdown_report']}")

    config = record.get("config")
    if config:
        for key in ("path", "source", "sha256"):
            if key not in config:
                raise ValueError(f"Experiment config missing {key}")
        _validate_hash(str(config["sha256"]), "config.sha256")
        if verify_artifacts:
            if config["source"] == "git_blob_at_commit":
                observed = _git_blob_sha256(
                    project_root,
                    str(record["git_commit"]),
                    str(config["path"]),
                )
            else:
                observed = sha256_file(
                    _resolve_project_file(project_root, str(config["path"]))
                )
            if observed != str(config["sha256"]):
                raise ValueError(f"Config SHA-256 mismatch for {record['id']}")

    if record.get("kind") == "formal_model":
        oof = record["oof"]
        if not isinstance(oof, Mapping) or not {"path", "sha256"} <= set(oof):
            raise ValueError("Formal model oof must contain path and sha256")
        _validate_hash(str(oof["sha256"]), "oof.sha256")
        if verify_artifacts:
            oof_path = _resolve_project_file(project_root, str(oof["path"]))
            if not oof_path.is_file() or sha256_file(oof_path) != str(oof["sha256"]):
                raise ValueError(f"OOF artifact mismatch for {record['id']}")
            result = record["results"]
            evaluation_path = _resolve_project_file(
                project_root,
                str(result["evaluation_path"]),
            )
            if (
                not evaluation_path.is_file()
                or sha256_file(evaluation_path) != str(result["evaluation_sha256"])
            ):
                raise ValueError(f"Evaluation artifact mismatch for {record['id']}")


def validate_research_ledger(
    ledger_path: Path,
    project_root: Path,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise ValueError("Unsupported research ledger schema")
    _validate_hash(str(ledger.get("data_fingerprint", "")), "ledger data_fingerprint")
    experiments = ledger.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Research ledger must contain experiments")
    ids = [record.get("id") for record in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("Research experiment IDs must be unique")
    for record in experiments:
        validate_experiment_record(record, project_root, verify_artifacts)
    if verify_artifacts:
        mlflow_path = _resolve_project_file(project_root, str(ledger["mlflow"]["database"]))
        if not mlflow_path.is_file():
            raise ValueError("MLflow database is missing")
    return {
        "study_id": ledger["study_id"],
        "experiments": len(experiments),
        "formal_models": sum(record["kind"] == "formal_model" for record in experiments),
    }
