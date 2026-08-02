"""Validate S1a clinical decisions without modifying the review configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree
import zipfile

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_REPORTS_DIR  # noqa: E402
from src.research_annotation_confirmation import (  # noqa: E402
    validate_completed_confirmation,
    validate_confirmation_template,
)
from src.research_ledger import sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402


DEFAULT_CONFIRMATION = (
    ROOT / "configs/research/annotation_clinical_confirmation_v0.yaml"
)
DEFAULT_REVIEW_CONFIG = ROOT / "configs/research/annotation_review_queue_v0.yaml"
DEFAULT_SOURCE_WORKBOOK = (
    PATIENT_MULTIMODAL_REPORTS_DIR
    / "annotation_review"
    / "S1a临床定义与复核参数确认表_v0_2026-08-03.xlsx"
)
OUTPUT_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
MAX_YAML_BYTES = 1_000_000
MAX_XLSX_BYTES = 20_000_000
MAX_WORKBOOK_XML_BYTES = 2_000_000
REQUIRED_WORKBOOK_SHEETS = {
    "填写说明",
    "医学定义确认",
    "复核参数确认",
    "字段代码本",
    "文献与版本",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--review-config", type=Path, default=DEFAULT_REVIEW_CONFIG)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    parser.add_argument(
        "--completed-workbook",
        type=Path,
        help="Returned, filled workbook. Required unless --template-check is used.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template-check", action="store_true")
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    if path.stat().st_size > MAX_YAML_BYTES:
        raise ValueError("Clinical confirmation YAML exceeds the size limit")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Clinical confirmation YAML must contain a mapping")
    return payload


def _verify_deviation_adr(payload: dict) -> dict[str, str] | None:
    reference = payload.get("deviation_decision_reference")
    if not reference:
        return None
    path = (ROOT / str(reference)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Deviation ADR escapes the project root") from error
    if not path.is_file():
        raise ValueError("Referenced deviation ADR does not exist")
    return {"path": str(reference), "sha256": sha256_file(path)}


def _validate_workbook_container(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_XLSX_BYTES:
        raise ValueError("Confirmation workbook exceeds the size limit")
    if not zipfile.is_zipfile(path):
        raise ValueError("Confirmation workbook is not a valid XLSX container")
    with zipfile.ZipFile(path) as archive:
        try:
            workbook_info = archive.getinfo("xl/workbook.xml")
        except KeyError as error:
            raise ValueError("Confirmation workbook has no workbook definition") from error
        if workbook_info.file_size > MAX_WORKBOOK_XML_BYTES:
            raise ValueError("Confirmation workbook definition exceeds the size limit")
        workbook_xml = archive.read(workbook_info)
    root = ElementTree.fromstring(workbook_xml)
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    observed = {
        str(element.attrib["name"])
        for element in root.findall(".//main:sheet", namespace)
    }
    missing = sorted(REQUIRED_WORKBOOK_SHEETS - observed)
    if missing:
        raise ValueError(
            "Confirmation workbook is missing required sheets: " + ", ".join(missing)
        )
    return {"xlsx_container_valid": True, "required_sheets_present": sorted(observed)}


def _project_output(path: Path | None, input_sha256: str) -> Path:
    if path is None:
        path = OUTPUT_DIR / (
            f"annotation_clinical_confirmation_validation_{input_sha256[:12]}.json"
        )
    resolved = path.resolve()
    try:
        resolved.relative_to(OUTPUT_DIR.resolve())
    except ValueError as error:
        raise ValueError("Validation output must stay inside the review report directory") from error
    if resolved.suffix.casefold() != ".json":
        raise ValueError("Validation output must be a JSON file")
    return resolved


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def main() -> int:
    args = parse_args()
    set_below_normal_priority()
    input_path = args.input.resolve()
    review_config_path = args.review_config.resolve()
    source_workbook_path = args.source_workbook.resolve()
    if not source_workbook_path.is_file():
        raise ValueError("Blank source workbook is missing")
    source_workbook_contract = _validate_workbook_container(source_workbook_path)
    payload = _read_yaml(input_path)
    review_config = _read_yaml(review_config_path)
    common = {
        "expected_dataset_fingerprint": review_config["dataset_fingerprint"],
        "expected_annotation_version": review_config["annotation_version"],
        "expected_source_workbook_sha256": sha256_file(source_workbook_path),
    }

    if args.template_check:
        if args.completed_workbook is not None:
            raise ValueError("--completed-workbook is not allowed with --template-check")
        result = validate_confirmation_template(payload, **common)
        result["provenance"] = {
            "confirmation_yaml_sha256": sha256_file(input_path),
            "review_config_sha256": sha256_file(review_config_path),
            "source_workbook_sha256": sha256_file(source_workbook_path),
            "dataset_fingerprint": review_config["dataset_fingerprint"],
            "annotation_version": review_config["annotation_version"],
            "workbook_contract": source_workbook_contract,
            "validation_git": _git_state(),
        }
        output_path = _project_output(args.output, sha256_file(input_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "ready_for_preregistration": False,
                    "report": output_path.relative_to(ROOT).as_posix(),
                    "report_sha256": sha256_file(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.completed_workbook is None:
        raise ValueError("--completed-workbook is required for strict validation")
    completed_workbook_path = args.completed_workbook.resolve()
    if not completed_workbook_path.is_file():
        raise ValueError("Completed workbook is missing")
    completed_workbook_contract = _validate_workbook_container(
        completed_workbook_path
    )
    completed_workbook_sha256 = sha256_file(completed_workbook_path)
    deviation_adr = _verify_deviation_adr(payload)
    result = validate_completed_confirmation(
        payload,
        expected_completed_workbook_sha256=completed_workbook_sha256,
        deviation_reference_verified=deviation_adr is not None,
        **common,
    )
    result["provenance"] = {
        "confirmation_yaml_sha256": sha256_file(input_path),
        "review_config_sha256": sha256_file(review_config_path),
        "source_workbook_sha256": sha256_file(source_workbook_path),
        "completed_workbook_sha256": completed_workbook_sha256,
        "dataset_fingerprint": review_config["dataset_fingerprint"],
        "annotation_version": review_config["annotation_version"],
        "deviation_adr": deviation_adr,
        "source_workbook_contract": source_workbook_contract,
        "completed_workbook_contract": completed_workbook_contract,
    }
    result["provenance"]["validation_git"] = _git_state()
    output_path = _project_output(args.output, sha256_file(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "ready_for_preregistration": result[
                    "ready_for_preregistration"
                ],
                "report": output_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["ready_for_preregistration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
