"""Open the draft clinical review queue in a diagnosis-blinded ROI-only window."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

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
from src.research_annotation_agreement import validate_review_template  # noqa: E402
from src.research_ledger import sha256_file  # noqa: E402
from src.research_annotation_review_ui import (  # noqa: E402
    ReviewImageRepository,
    ReviewQueuePreviewWindow,
    audit_review_queue_rois,
    configure_cjk_font,
)
from src.research_runtime import set_below_normal_priority  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/annotation_review_queue_v0.yaml",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=(
            PATIENT_MULTIMODAL_REPORTS_DIR
            / "annotation_review"
            / "annotation_review_queue_draft.csv"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the first ROI offscreen and exit without entering the event loop.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Optional project-local screenshot path for offscreen UI verification.",
    )
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="Load all queue ROIs and write a privacy-safe aggregate audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test and args.audit_all:
        raise ValueError("Choose either --smoke-test or --audit-all")
    started = time.perf_counter()
    set_below_normal_priority()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    rows = _read_csv(args.queue.resolve())
    contract = validate_review_template(rows, set(config["review_targets"]))
    source_rows = _read_csv(
        PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
    )
    repository = ReviewImageRepository(ROOT, source_rows)
    if args.audit_all:
        result = audit_review_queue_rois(rows, repository)
        result["status"] = (
            "PASS" if result["failed_rois"] == 0 else "FAIL"
        )
        result["runtime_seconds"] = time.perf_counter() - started
        result["provenance"] = {
            "queue_sha256": sha256_file(args.queue.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "dataset_fingerprint": config["dataset_fingerprint"],
            "annotation_version": config["annotation_version"],
        }
        output_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "annotation_review_roi_load_audit.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "loaded_rois": result["loaded_rois"],
                    "failed_rois": result["failed_rois"],
                    "report": output_path.relative_to(ROOT).as_posix(),
                    "report_sha256": sha256_file(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["failed_rois"] == 0 else 1
    app = QApplication.instance() or QApplication([])
    cjk_font = configure_cjk_font(app)
    window = ReviewQueuePreviewWindow(rows, repository)
    if args.smoke_test:
        window.show()
        app.processEvents()
        if window._source_pixmap.isNull():
            raise RuntimeError("First review ROI failed to load")
        screenshot = None
        if args.screenshot:
            screenshot_path = args.screenshot.resolve()
            try:
                screenshot_path.relative_to(ROOT)
            except ValueError as error:
                raise ValueError("Screenshot path must stay inside the project") from error
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(screenshot_path)):
                raise RuntimeError("Could not save UI verification screenshot")
            screenshot = screenshot_path.relative_to(ROOT).as_posix()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "draft_read_only_preview",
                    "rows": contract["rows"],
                    "first_roi_loaded": True,
                    "diagnosis_or_legacy_overlay_visible": False,
                    "cjk_font_loaded": cjk_font is not None,
                    "screenshot": screenshot,
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
