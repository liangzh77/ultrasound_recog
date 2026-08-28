"""Prepare sparse class-specific contour masks for the M0--M3 study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_semantic_contours import (  # noqa: E402
    render_category_masks,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "configs/research/m_series_semantic_contour_preparation.yaml"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_project_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Configured path must be project-relative")
    candidate = (ROOT / relative).resolve()
    candidate.relative_to(ROOT.resolve())
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_preregistered_retrospective_preparation":
        raise ValueError("M-series semantic contour preparation must be frozen")
    categories = tuple(config.get("finding_categories") or ())
    if len(categories) != 13 or len(set(categories)) != 13:
        raise ValueError("M-series requires the 13 frozen finding categories")
    output_size = int(config["input_contract"]["output_size"])
    images = {
        row["image_key"]: row
        for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv")
        if row["include"] == "1"
    }
    sources = {
        row["image_key"]: row
        for row in _read_csv(
            PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
        )
    }
    if set(images) - set(sources):
        raise ValueError("Included images are missing normalized annotation sources")
    if len(images) != int(config["cohort"]["expected_images"]):
        raise ValueError("M-series image count differs from the frozen cohort")

    mask_root = _safe_project_path(config["outputs"]["mask_directory"])
    manifest_path = _safe_project_path(config["outputs"]["manifest"])
    summary_path = _safe_project_path(config["outputs"]["summary"])
    mask_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    category_objects: Counter[str] = Counter()
    category_images: Counter[str] = Counter()
    category_patients: dict[str, set[str]] = {category: set() for category in categories}
    category_pixels: Counter[str] = Counter()
    positive_images = 0
    for image_key in sorted(images):
        image_row = images[image_key]
        source_path = _safe_project_path(
            sources[image_key]["normalized_annotation_path"]
        )
        annotation = json.loads(source_path.read_text(encoding="utf-8"))
        roi = {
            name: float(image_row[f"roi_{name}"])
            for name in ("x1", "y1", "x2", "y2")
        }
        masks, counts, pixels = render_category_masks(
            annotation, roi, categories, output_size
        )
        sparse: dict[str, dict[str, str | int]] = {}
        for category in categories:
            if counts[category] < 1 or pixels[category] < 1:
                continue
            category_dir = mask_root / f"category_{categories.index(category):02d}"
            category_dir.mkdir(parents=True, exist_ok=True)
            destination = category_dir / f"{image_key}.png"
            masks[category].save(destination, format="PNG", optimize=True)
            sparse[category] = {
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(destination),
                "object_count": counts[category],
                "positive_pixels": pixels[category],
            }
            category_objects[category] += counts[category]
            category_images[category] += 1
            category_pixels[category] += pixels[category]
            category_patients[category].add(image_row["person_key"])
        positive_images += bool(sparse)
        rows.append(
            {
                "image_key": image_key,
                "category_masks_json": json.dumps(
                    sparse, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image_key", "category_masks_json")
        )
        writer.writeheader()
        writer.writerows(rows)
    patients = {row["person_key"] for row in images.values()}
    if len(patients) != int(config["cohort"]["expected_patients"]):
        raise ValueError("M-series patient count differs from the frozen cohort")
    summary = {
        "status": "completed_retrospective_preparation",
        "study_code": "M0-M3",
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "patients": len(patients),
        "images": len(rows),
        "positive_contour_images": positive_images,
        "all_zero_contour_images": len(rows) - positive_images,
        "categories": [
            {
                "category": category,
                "objects": category_objects[category],
                "images": category_images[category],
                "patients": len(category_patients[category]),
                "positive_pixels": category_pixels[category],
            }
            for category in categories
        ],
        "category_order": list(categories),
        "input_channels_for_m2": 3 + len(categories),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "sparse_mask_files": sum(category_images.values()),
        "complete_rgb_preserved": True,
        "overlapping_categories_preserved": True,
        "contains_diagnosis": False,
        "contains_raw_paths": False,
        "allow_2026": False,
        "absence_interpretation": (
            "historical_annotation_not_observed_not_confirmed_medical_negative"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
