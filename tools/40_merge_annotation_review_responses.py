"""Merge two complete isolated S1a responses for a later adjudication stage."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_REPORTS_DIR  # noqa: E402
from src.research_annotation_review_entry import (  # noqa: E402
    merge_independent_reviewer_rows,
    validate_formal_entry_config,
    write_response_csv_atomic,
)
from src.research_ledger import sha256_file  # noqa: E402


REVIEW_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
DEFAULT_CONFIG = ROOT / "configs/research/annotation_review_queue_v0.yaml"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _controlled_path(path: Path, expected_name: str) -> Path:
    resolved = path.resolve()
    if resolved.parent != REVIEW_DIR.resolve() or resolved.name != expected_name:
        raise ValueError("Review artifact must use its controlled filename")
    return resolved


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Independent response has no provenance manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_source_manifest(
    manifest: dict[str, Any],
    path: Path,
    slot: int,
    config_sha256: str,
) -> str:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study")
        != "knee_annotation_review_s1a_independent_response"
        or manifest.get("status") != "complete"
        or manifest.get("reviewer_slot") != slot
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("response_sha256") != sha256_file(path)
    ):
        raise ValueError("Independent response manifest is incomplete or mismatched")
    queue_sha256 = str(manifest.get("queue_sha256", ""))
    if len(queue_sha256) != 64:
        raise ValueError("Independent response queue hash is invalid")
    return queue_sha256


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reviewer-1",
        type=Path,
        default=REVIEW_DIR / "annotation_review_reviewer_1_response.csv",
    )
    parser.add_argument(
        "--reviewer-2",
        type=Path,
        default=REVIEW_DIR / "annotation_review_reviewer_2_response.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REVIEW_DIR / "annotation_review_merged_for_adjudication.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if (
        config_path.parent != (ROOT / "configs/research").resolve()
        or not config_path.match("annotation_review_queue_v*.yaml")
    ):
        raise ValueError("Merge config must use a controlled project artifact")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_entry_config(config)
    git_state = _git_state()
    if git_state["dirty"] or git_state["commit"] != config["frozen_provenance"][
        "preregistration_git_commit"
    ]:
        raise ValueError("Merge runtime Git differs from preregistration")
    config_sha256 = sha256_file(config_path)
    reviewer_1_path = _controlled_path(
        args.reviewer_1, "annotation_review_reviewer_1_response.csv"
    )
    reviewer_2_path = _controlled_path(
        args.reviewer_2, "annotation_review_reviewer_2_response.csv"
    )
    output_path = _controlled_path(
        args.output, "annotation_review_merged_for_adjudication.csv"
    )
    queue_1 = _validate_source_manifest(
        _manifest(reviewer_1_path), reviewer_1_path, 1, config_sha256
    )
    queue_2 = _validate_source_manifest(
        _manifest(reviewer_2_path), reviewer_2_path, 2, config_sha256
    )
    if queue_1 != queue_2:
        raise ValueError("Independent responses were created from different queues")
    merged = merge_independent_reviewer_rows(
        _read_csv(reviewer_1_path), _read_csv(reviewer_2_path), config
    )
    write_response_csv_atomic(output_path, merged)
    output_manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_merged_for_adjudication",
        "status": "awaiting_blinded_adjudication",
        "config_sha256": config_sha256,
        "queue_sha256": queue_1,
        "reviewer_1_response_sha256": sha256_file(reviewer_1_path),
        "reviewer_2_response_sha256": sha256_file(reviewer_2_path),
        "merged_response_sha256": sha256_file(output_path),
        "merge_git": git_state,
        "rows": len(merged),
        "privacy": {
            "reviewer_names_recorded": False,
            "diagnosis_present": False,
            "legacy_annotation_present": False,
            "raw_paths_present": False,
        },
        "geometry": {"captured_in_s1a": False, "deferred_stage": "S1b"},
    }
    _write_json_atomic(output_path.with_suffix(".manifest.json"), output_manifest)
    print(
        json.dumps(
            {
                "status": output_manifest["status"],
                "rows": len(merged),
                "output": output_path.relative_to(ROOT).as_posix(),
                "output_sha256": output_manifest["merged_response_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
