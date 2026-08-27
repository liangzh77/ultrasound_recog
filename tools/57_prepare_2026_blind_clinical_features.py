"""Prepare private C3 features from the diagnosis-hidden 2026 workbook."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_REGISTRY_DIR, PATIENT_MULTIMODAL_REPORTS_DIR  # noqa: E402
from src.research_auto_roi import require_sha256, sha256_file  # noqa: E402
from src.research_clinical import (  # noqa: E402
    ALLOWED_MODEL_FEATURES,
    BLIND_SOURCE_COLUMN_INDEX,
    CENSOR_FEATURES,
    MISSING_FEATURES,
    load_clinical_config,
    parse_clinical_model_features,
    validate_blind_source_headers,
)


R2_CONFIG_PATH = ROOT / "configs/research/r2_2026_frozen_inference.yaml"
C3_CONFIG_PATH = ROOT / "configs/research/c0_c4_clinical_logreg.yaml"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    r2_config = yaml.safe_load(R2_CONFIG_PATH.read_text(encoding="utf-8"))
    clinical_config = load_clinical_config(C3_CONFIG_PATH)
    cohort = r2_config["cohort"]
    workbook_path = (ROOT / cohort["clinical_workbook"]).resolve()
    workbook_hash = require_sha256(
        workbook_path,
        cohort["clinical_workbook_sha256"],
        "2026 clinical workbook",
    )
    private_dir = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "r2_2026"
    people_path = private_dir / "2026_person_identity_map.csv"
    people = read_csv(people_path)
    person_by_number = {
        int(row["hospital_patient_number"]): row["person_key"] for row in people
    }
    expected_numbers = set(range(1, int(cohort["patients"]) + 1))
    if set(person_by_number) != expected_numbers:
        raise ValueError("private 2026 person map does not cover identifiers 1 through 225")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if len(workbook.sheetnames) != 1:
        raise ValueError("2026 clinical workbook must contain exactly one sheet")
    sheet = workbook[workbook.sheetnames[0]]
    headers = tuple(
        "" if cell.value is None else str(cell.value).strip()
        for cell in next(sheet.iter_rows(min_row=1, max_row=1))
    )
    validate_blind_source_headers(headers)

    rows = []
    observed_numbers = set()
    missing_counts: Counter[str] = Counter()
    censored_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    for source_row in sheet.iter_rows(min_row=2, values_only=True):
        if not source_row or all(value is None for value in source_row):
            continue
        if len(source_row) != len(headers):
            raise ValueError("2026 clinical row width differs from the frozen header")
        source = {
            feature: source_row[index]
            for feature, index in BLIND_SOURCE_COLUMN_INDEX.items()
        }
        raw_number = source.pop("patient_number")
        if isinstance(raw_number, bool):
            raise ValueError("2026 patient number must be an integer")
        try:
            patient_number = int(raw_number)
        except (TypeError, ValueError) as error:
            raise ValueError("2026 patient number must be an integer") from error
        if patient_number not in expected_numbers or patient_number in observed_numbers:
            raise ValueError("2026 clinical patient number is missing, duplicated, or outside range")
        observed_numbers.add(patient_number)
        features, audit = parse_clinical_model_features(source, clinical_config)
        missing_counts.update(audit["missing_features"])
        censored_counts.update(audit["censored_features"])
        invalid_counts.update(audit["invalid_features"])
        rows.append({"person_key": person_by_number[patient_number], **features})
    workbook.close()

    if observed_numbers != expected_numbers or len(rows) != int(cohort["patients"]):
        raise ValueError("2026 clinical workbook does not cover every frozen patient exactly once")
    rows.sort(key=lambda row: str(row["person_key"]))
    fields = ("person_key", *ALLOWED_MODEL_FEATURES, *MISSING_FEATURES, *CENSOR_FEATURES)
    output_path = private_dir / "2026_clinical_features.csv"
    write_csv(output_path, rows, fields)

    summary = {
        "experiment_code": "R2",
        "status": "blind_clinical_features_prepared",
        "patients": len(rows),
        "source_workbook_sha256": workbook_hash,
        "private_person_map_sha256": sha256_file(people_path),
        "private_feature_table_sha256": sha256_file(output_path),
        "model_feature_whitelist": list(ALLOWED_MODEL_FEATURES),
        "forbidden_duration_excluded": True,
        "explicit_missingness_features_excluded_from_C3": True,
        "missing_counts": dict(sorted(missing_counts.items())),
        "censored_boundary_counts": dict(sorted(censored_counts.items())),
        "nonnumeric_values_set_missing": dict(sorted(invalid_counts.items())),
        "public_contains_patient_values": False,
        "reference_diagnoses_read": False,
        "formal_prediction_generated": False,
    }
    report_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "2026_clinical_feature_audit.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(report_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

