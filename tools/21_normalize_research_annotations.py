"""Generate a versioned 28-class annotation set without copying images."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_DERIVED_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
    RAW_LABEL_DIR,
)
from src.label_mapping import get_disease_from_label  # noqa: E402
from src.research_annotations import load_and_normalize  # noqa: E402
from src.research_sources import IMAGE_EXTENSIONS  # noqa: E402


EXPECTED_CATEGORY_COUNT = 28


def main() -> int:
    freeze_path = PATIENT_MULTIMODAL_REPORTS_DIR / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    version_short = freeze["dataset_version_short"]
    target = PATIENT_MULTIMODAL_DERIVED_DIR / f"annotations_{version_short}"
    if target.exists():
        print(target)
        print("Existing version retained; no files were overwritten.")
        return 0

    PATIENT_MULTIMODAL_DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".annotations_{version_short}_",
            dir=PATIENT_MULTIMODAL_DERIVED_DIR,
        )
    )
    annotations_root = staging / "annotations"
    raw_categories = Counter()
    normalized_categories = Counter()
    mapping_counts = Counter()
    summary = Counter()

    for disease_dir in sorted(path for path in RAW_LABEL_DIR.iterdir() if path.is_dir()):
        for patient_dir in sorted(path for path in disease_dir.iterdir() if path.is_dir()):
            images = {
                path.stem
                for path in patient_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            jsons = {path.stem: path for path in patient_dir.glob("*.json")}
            summary["missing_annotation_images"] += len(images - jsons.keys())
            summary["orphan_annotations"] += len(jsons.keys() - images)
            for stem in sorted(images & jsons.keys()):
                normalized, changes, object_count = load_and_normalize(
                    jsons[stem],
                    disease_dir.name,
                )
                output = (
                    annotations_root
                    / disease_dir.name
                    / patient_dir.name
                    / f"{stem}.json"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary["annotation_files"] += 1
                summary["objects_before"] += object_count
                summary["objects_after"] += len(normalized.get("objects", []))
                for item in normalized.get("objects", []):
                    normalized_categories[item["category"]] += 1
                source = json.loads(jsons[stem].read_text(encoding="utf-8"))
                for item in source.get("objects", []):
                    raw_categories[str(item.get("category", ""))] += 1
                for old, new in changes.items():
                    mapping_counts[(old, new)] += 1

    residual_prefixes = sorted(
        category
        for category in normalized_categories
        if get_disease_from_label(category) is not None
    )
    if len(normalized_categories) != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORY_COUNT} categories, "
            f"found {len(normalized_categories)}"
        )
    if residual_prefixes:
        raise ValueError(f"Residual disease prefixes: {residual_prefixes}")
    if summary["objects_before"] != summary["objects_after"]:
        raise ValueError("Object count changed during normalization")

    category_mapping = {
        "dataset_version": freeze["dataset_version"],
        "categories": sorted(normalized_categories),
        "category_to_id": {
            category: index
            for index, category in enumerate(sorted(normalized_categories))
        },
    }
    (staging / "category_mapping.json").write_text(
        json.dumps(category_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_version": freeze["dataset_version"],
        "annotation_version": target.name,
        **dict(summary),
        "raw_category_count": len(raw_categories),
        "normalized_category_count": len(normalized_categories),
        "residual_disease_prefixes": residual_prefixes,
        "images_copied": 0,
        "raw_annotations_modified": 0,
        "categories": dict(normalized_categories.most_common()),
        "mapping": [
            {"raw": old, "normalized": new, "files_or_groups": count}
            for (old, new), count in sorted(mapping_counts.items())
        ],
    }
    (staging / "normalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(staging, target)
    (PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_normalization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(target)
    print(json.dumps({key: report[key] for key in (
        "annotation_files",
        "objects_before",
        "normalized_category_count",
        "residual_disease_prefixes",
        "images_copied",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
