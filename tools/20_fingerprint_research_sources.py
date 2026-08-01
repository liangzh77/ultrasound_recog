"""Freeze development and 2026 blind source files with SHA-256."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
    RAW_DATA_DIR,
    RAW_LABEL_DIR,
)
from src.research_schema import (  # noqa: E402
    DIAGNOSIS_CLASSES,
    DIAGNOSIS_TO_ID,
    EXCLUDED_DIAGNOSES,
)
from src.research_sources import (  # noqa: E402
    dataset_version,
    fingerprint_tree,
)


BLIND_2026_DIR = RAW_DATA_DIR / "膝关节2026未标注"


def main() -> int:
    if not RAW_LABEL_DIR.is_dir():
        raise FileNotFoundError(RAW_LABEL_DIR)
    if not BLIND_2026_DIR.is_dir():
        raise FileNotFoundError(BLIND_2026_DIR)

    records = fingerprint_tree(
        RAW_LABEL_DIR,
        cohort="development_labeled",
        relative_to=RAW_DATA_DIR,
    )
    records.extend(
        fingerprint_tree(
            BLIND_2026_DIR,
            cohort="blind_2026",
            relative_to=RAW_DATA_DIR,
        )
    )
    version = dataset_version(records)

    PATIENT_MULTIMODAL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    PATIENT_MULTIMODAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PATIENT_MULTIMODAL_REGISTRY_DIR / "source_files.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort",
                "relative_path",
                "kind",
                "size_bytes",
                "mtime_ns",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)

    counts = Counter((record.cohort, record.kind) for record in records)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_version": version,
        "dataset_version_short": version[:12],
        "source_file_count": len(records),
        "source_bytes": sum(record.size_bytes for record in records),
        "counts": {
            f"{cohort}:{kind}": count
            for (cohort, kind), count in sorted(counts.items())
        },
        "private_file_manifest": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "blind_2026_policy": (
            "Inputs are fingerprinted only. Labels are unavailable and this "
            "cohort must not be used for model selection."
        ),
    }
    (PATIENT_MULTIMODAL_REPORTS_DIR / "source_freeze.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reference = {
        "task": "patient_level_primary_diagnosis",
        "classes": list(DIAGNOSIS_CLASSES),
        "class_to_id": DIAGNOSIS_TO_ID,
        "excluded_primary_diagnoses": sorted(EXCLUDED_DIAGNOSES),
        "development_reference_standard": (
            "Disease directory and audited Excel diagnosis are reference-only; "
            "diagnosis text is never a model feature."
        ),
        "coexisting_disease_policy": (
            "The current target is the recorded primary diagnosis, not a "
            "strictly mutually-exclusive multi-disease truth."
        ),
    }
    (PATIENT_MULTIMODAL_REGISTRY_DIR / "reference_standard.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
