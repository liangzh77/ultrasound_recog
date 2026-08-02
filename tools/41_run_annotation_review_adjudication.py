"""Run diagnosis-blinded S1a disagreement adjudication after response merging."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

if "--smoke-test" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_annotation_adjudication_ui import (  # noqa: E402
    ReviewAdjudicationWindow,
)
from src.research_annotation_review_entry import (  # noqa: E402
    new_adjudication_rows,
    validate_adjudication_rows,
    validate_formal_entry_config,
    write_response_csv_atomic,
)
from src.research_annotation_review_ui import (  # noqa: E402
    ReviewImageRepository,
    configure_cjk_font,
)
from src.research_ledger import sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402


REVIEW_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
DEFAULT_CONFIG = ROOT / "configs/research/annotation_review_queue_v0.yaml"
DEFAULT_MERGED = REVIEW_DIR / "annotation_review_merged_for_adjudication.csv"
DEFAULT_OUTPUT = REVIEW_DIR / "annotation_review_adjudicated.csv"
RUNTIME_CODE_PATHS = (
    "src/research_annotation_review.py",
    "src/research_annotation_review_entry.py",
    "src/research_annotation_adjudication_ui.py",
    "src/research_annotation_review_ui.py",
    "tools/41_run_annotation_review_adjudication.py",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _controlled_path(path: Path, parent: Path, expected_name: str) -> Path:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != expected_name:
        raise ValueError("Adjudication artifact must use its controlled filename")
    return resolved


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


def _validate_runtime_git(expected_code_commit: str) -> dict[str, Any]:
    state = _git_state()
    if state["dirty"]:
        raise ValueError("Adjudication runtime Git must be clean")
    comparison = subprocess.run(
        ["git", "diff", "--quiet", expected_code_commit, "--", *RUNTIME_CODE_PATHS],
        cwd=ROOT,
        check=False,
    )
    if comparison.returncode != 0:
        raise ValueError("Adjudication code differs from the frozen workflow Git")
    return state


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


def _validate_merged_manifest(
    manifest: dict[str, Any],
    merged_path: Path,
    config_sha256: str,
    queue_sha256: str,
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study")
        != "knee_annotation_review_s1a_merged_for_adjudication"
        or manifest.get("status") != "awaiting_blinded_adjudication"
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("queue_sha256") != queue_sha256
        or manifest.get("merged_response_sha256") != sha256_file(merged_path)
    ):
        raise ValueError("Merged review manifest is missing or mismatched")


def _validate_resume_manifest(
    manifest: dict[str, Any],
    output_path: Path,
    config_sha256: str,
    queue_sha256: str,
    merged_sha256: str,
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study")
        != "knee_annotation_review_s1a_blinded_adjudication"
        or manifest.get("status") not in {"in_progress", "complete"}
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("queue_sha256") != queue_sha256
        or manifest.get("merged_response_sha256") != merged_sha256
        or manifest.get("adjudicated_response_sha256") != sha256_file(output_path)
        or manifest.get("privacy", {}).get("adjudicator_name_recorded") is not False
    ):
        raise ValueError("Existing adjudication manifest is mismatched")


def _save_adjudication(
    output_path: Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    config_sha256: str,
    merged_sha256: str,
    started: float,
) -> dict[str, Any]:
    progress = validate_adjudication_rows(rows, config, require_complete=False)
    git_state = _validate_runtime_git(
        config["frozen_provenance"]["review_workflow_git_commit"]
    )
    write_response_csv_atomic(output_path, rows)
    manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_blinded_adjudication",
        "status": (
            "complete" if progress["remaining_rows"] == 0 else "in_progress"
        ),
        "config_sha256": config_sha256,
        "queue_sha256": config["frozen_provenance"]["queue_sha256"],
        "merged_response_sha256": merged_sha256,
        "adjudicated_response_sha256": sha256_file(output_path),
        "source_git": git_state,
        "runtime_seconds_current_session": time.perf_counter() - started,
        "progress": progress,
        "privacy": {
            "adjudicator_name_recorded": False,
            "reviewer_names_recorded": False,
            "diagnosis_visible": False,
            "legacy_annotation_visible": False,
            "raw_paths_recorded": False,
        },
        "geometry": {"captured_in_s1a": False, "deferred_stage": "S1b"},
    }
    _write_json_atomic(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the first adjudication ROI offscreen and exit without saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    set_below_normal_priority()
    config_path = args.config.resolve()
    if (
        config_path.parent != (ROOT / "configs/research").resolve()
        or not config_path.match("annotation_review_queue_v*.yaml")
    ):
        raise ValueError("Adjudication config must use a controlled project artifact")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_entry_config(config)
    _validate_runtime_git(config["frozen_provenance"]["review_workflow_git_commit"])
    merged_path = _controlled_path(
        args.merged, REVIEW_DIR, "annotation_review_merged_for_adjudication.csv"
    )
    output_path = _controlled_path(
        args.output, REVIEW_DIR, "annotation_review_adjudicated.csv"
    )
    config_sha256 = sha256_file(config_path)
    merged_manifest_path = merged_path.with_suffix(".manifest.json")
    if not merged_manifest_path.is_file():
        raise ValueError("Merged review has no provenance manifest")
    _validate_merged_manifest(
        json.loads(merged_manifest_path.read_text(encoding="utf-8")),
        merged_path,
        config_sha256,
        config["frozen_provenance"]["queue_sha256"],
    )
    merged_sha256 = sha256_file(merged_path)
    if output_path.exists():
        manifest_path = output_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError("Existing adjudication has no provenance manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(
            manifest,
            output_path,
            config_sha256,
            config["frozen_provenance"]["queue_sha256"],
            merged_sha256,
        )
        rows = _read_csv(output_path)
        validate_adjudication_rows(rows, config, require_complete=False)
    else:
        rows = new_adjudication_rows(_read_csv(merged_path), config)

    source_rows = _read_csv(
        PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
    )
    repository = ReviewImageRepository(ROOT, source_rows)
    app = QApplication.instance() or QApplication([])
    configure_cjk_font(app)
    window = ReviewAdjudicationWindow(
        rows,
        config,
        repository,
        lambda value: _save_adjudication(
            output_path,
            value,
            config,
            config_sha256,
            merged_sha256,
            started,
        ),
    )
    if args.smoke_test:
        window.show()
        app.processEvents()
        if window._source_pixmap.isNull():
            raise RuntimeError("First adjudication ROI failed to load")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "blinded_adjudication_smoke_test_no_write",
                    "rows": len(rows),
                    "output_written": False,
                    "diagnosis_or_legacy_overlay_visible": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        window.close()
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
