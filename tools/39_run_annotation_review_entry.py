"""Run one diagnosis-blinded S1a reviewer entry session after preregistration."""

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
from src.research_annotation_review_entry import (  # noqa: E402
    new_reviewer_response_rows,
    validate_formal_entry_config,
    validate_reviewer_response_rows,
    write_response_csv_atomic,
)
from src.research_annotation_review_entry_ui import (  # noqa: E402
    ReviewQueueEntryWindow,
)
from src.research_annotation_review_ui import (  # noqa: E402
    ReviewImageRepository,
    configure_cjk_font,
)
from src.research_ledger import sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402


REVIEW_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
DEFAULT_CONFIG = ROOT / "configs/research/annotation_review_queue_v0.yaml"
DEFAULT_QUEUE = REVIEW_DIR / "annotation_review_queue_draft.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _controlled_response_path(requested: Path | None, slot: int) -> Path:
    expected_name = f"annotation_review_reviewer_{slot}_response.csv"
    path = (requested or REVIEW_DIR / expected_name).resolve()
    if path.parent != REVIEW_DIR.resolve() or path.name != expected_name:
        raise ValueError("Reviewer response must use the controlled review filename")
    return path


def _controlled_input_path(path: Path, parent: Path, pattern: str) -> Path:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.match(pattern):
        raise ValueError("Formal review input must use a controlled project artifact")
    return resolved


def _manifest_path(response_path: Path) -> Path:
    return response_path.with_suffix(".manifest.json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_resume_manifest(
    manifest: dict[str, Any],
    response_path: Path,
    config_sha256: str,
    queue_sha256: str,
    slot: int,
) -> None:
    expected = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_independent_response",
        "reviewer_slot": slot,
        "config_sha256": config_sha256,
        "queue_sha256": queue_sha256,
        "response_sha256": sha256_file(response_path),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Reviewer response manifest does not match the current session")
    if manifest.get("privacy", {}).get("reviewer_name_recorded") is not False:
        raise ValueError("Reviewer response manifest violates the privacy contract")


def _save_session(
    response_path: Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    slot: int,
    config_sha256: str,
    queue_sha256: str,
    started: float,
) -> dict[str, Any]:
    progress = validate_reviewer_response_rows(
        rows, config, slot, require_complete=False
    )
    git_state = _git_state()
    if git_state["dirty"] or git_state["commit"] != config["frozen_provenance"][
        "preregistration_git_commit"
    ]:
        raise ValueError("Formal review runtime Git differs from preregistration")
    write_response_csv_atomic(response_path, rows)
    manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_independent_response",
        "status": (
            "complete" if progress["remaining_rows"] == 0 else "in_progress"
        ),
        "reviewer_slot": slot,
        "config_sha256": config_sha256,
        "queue_sha256": queue_sha256,
        "response_sha256": sha256_file(response_path),
        "source_git": git_state,
        "runtime_seconds_current_session": time.perf_counter() - started,
        "progress": progress,
        "privacy": {
            "reviewer_name_recorded": False,
            "diagnosis_visible": False,
            "legacy_annotation_visible": False,
            "raw_paths_recorded": False,
            "other_reviewer_values_present": False,
        },
        "geometry": {
            "captured_in_s1a": False,
            "deferred_stage": "S1b",
        },
    }
    _write_json_atomic(_manifest_path(response_path), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-slot", type=int, choices=(1, 2), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--response", type=Path)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the first formal ROI offscreen and exit without saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    set_below_normal_priority()
    config_path = _controlled_input_path(
        args.config, ROOT / "configs/research", "annotation_review_queue_v*.yaml"
    )
    queue_path = _controlled_input_path(
        args.queue, REVIEW_DIR, "annotation_review_queue_*.csv"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_entry_config(config)
    response_path = _controlled_response_path(args.response, args.reviewer_slot)
    config_sha256 = sha256_file(config_path)
    queue_sha256 = sha256_file(queue_path)
    if queue_sha256 != config["frozen_provenance"]["queue_sha256"]:
        raise ValueError("Formal review queue hash differs from frozen provenance")
    runtime_git = _git_state()
    if runtime_git["dirty"] or runtime_git["commit"] != config[
        "frozen_provenance"
    ]["preregistration_git_commit"]:
        raise ValueError("Formal review runtime Git differs from preregistration")
    queue_rows = _read_csv(queue_path)

    if response_path.exists():
        manifest_path = _manifest_path(response_path)
        if not manifest_path.is_file():
            raise ValueError("Existing reviewer response has no provenance manifest")
        rows = _read_csv(response_path)
        validate_reviewer_response_rows(
            rows, config, args.reviewer_slot, require_complete=False
        )
        _validate_resume_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            response_path,
            config_sha256,
            queue_sha256,
            args.reviewer_slot,
        )
    else:
        rows = new_reviewer_response_rows(queue_rows, config, args.reviewer_slot)

    source_rows = _read_csv(
        PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
    )
    repository = ReviewImageRepository(ROOT, source_rows)
    app = QApplication.instance() or QApplication([])
    configure_cjk_font(app)
    window = ReviewQueueEntryWindow(
        rows,
        config,
        args.reviewer_slot,
        repository,
        lambda value: _save_session(
            response_path,
            value,
            config,
            args.reviewer_slot,
            config_sha256,
            queue_sha256,
            started,
        ),
    )
    if args.smoke_test:
        window.show()
        app.processEvents()
        if window._source_pixmap.isNull():
            raise RuntimeError("First formal review ROI failed to load")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "formal_entry_smoke_test_no_write",
                    "reviewer_slot": args.reviewer_slot,
                    "rows": len(rows),
                    "response_written": False,
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
