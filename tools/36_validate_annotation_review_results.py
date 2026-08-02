"""Validate a blinded review CSV and calculate preregistered agreement statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_REPORTS_DIR  # noqa: E402
from src.research_annotation_agreement import (  # noqa: E402
    validate_and_summarize_completed_review,
    validate_review_template,
)
from src.research_ledger import sha256_file  # noqa: E402


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
        "--input",
        type=Path,
        default=(
            PATIENT_MULTIMODAL_REPORTS_DIR
            / "annotation_review"
            / "annotation_review_queue_draft.csv"
        ),
    )
    parser.add_argument("--template-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    input_path = args.input.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _read_csv(input_path)
    if args.template_check:
        result = validate_review_template(rows, set(config["review_targets"]))
        result["status"] = "blank_template_contract_passed"
        result["input_sha256"] = sha256_file(input_path)
        result["config_sha256"] = sha256_file(config_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if config["status"] != "frozen_preregistered":
        raise ValueError(
            "Strict agreement analysis requires a frozen_preregistered config"
        )
    result = validate_and_summarize_completed_review(rows, config)
    result["provenance"] = {
        "input_sha256": sha256_file(input_path),
        "config_sha256": sha256_file(config_path),
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
    }
    output_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
    output_path = output_dir / "annotation_review_agreement.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": output_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
