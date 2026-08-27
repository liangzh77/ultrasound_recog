"""Build co-registered binary finding masks for the L0 patient study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_DERIVED_DIR,
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_lesion_guided import render_binary_finding_mask  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research/l0_lesion_mask_preparation.yaml"
MASK_DIR = PATIENT_MULTIMODAL_DERIVED_DIR / "L0_manual_lesion_masks_384"
REPORT_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "lesion_guided"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_project_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Source path must be project-relative")
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
    if config.get("status") != "frozen":
        raise ValueError("L0 mask preparation config must be frozen")
    if config.get("annotation_version") != "annotations_62ecb01c4d77":
        raise ValueError("Unexpected normalized annotation version")
    included_categories = set(config.get("included_categories") or [])
    if len(included_categories) != 13:
        raise ValueError("L0 v1 requires the 13 preregistered finding categories")
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

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    total_by_diagnosis: Counter[str] = Counter()
    positive_by_diagnosis: Counter[str] = Counter()
    zero_by_diagnosis: Counter[str] = Counter()
    patient_has_positive_mask: dict[str, bool] = {}
    zero_mask_images = 0
    total_finding_objects = 0
    for image_key in sorted(images):
        image_row = images[image_key]
        diagnosis = image_row["diagnosis"]
        person_key = image_row["person_key"]
        total_by_diagnosis[diagnosis] += 1
        patient_has_positive_mask.setdefault(person_key, False)
        annotation_path = _safe_project_path(
            sources[image_key]["normalized_annotation_path"]
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        roi = {
            name: float(image_row[f"roi_{name}"])
            for name in ("x1", "y1", "x2", "y2")
        }
        mask, object_count, positive_pixels = render_binary_finding_mask(
            annotation,
            roi,
            included_categories,
            output_size,
        )
        destination = MASK_DIR / f"{image_key}.png"
        mask.save(destination, format="PNG", optimize=True)
        total_finding_objects += object_count
        if positive_pixels == 0:
            zero_mask_images += 1
            zero_by_diagnosis[diagnosis] += 1
        else:
            positive_by_diagnosis[diagnosis] += 1
            patient_has_positive_mask[person_key] = True
        rows.append(
            {
                "image_key": image_key,
                "mask_path": destination.relative_to(ROOT).as_posix(),
                "mask_sha256": _sha256(destination),
                "finding_object_count": object_count,
                "positive_pixels": positive_pixels,
            }
        )

    manifest_path = REPORT_DIR / "L0_manual_lesion_mask_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mask_presence_by_diagnosis = {
        diagnosis: {
            "images": total_by_diagnosis[diagnosis],
            "positive_mask_images": positive_by_diagnosis[diagnosis],
            "zero_mask_images": zero_by_diagnosis[diagnosis],
            "positive_mask_rate": (
                positive_by_diagnosis[diagnosis] / total_by_diagnosis[diagnosis]
            ),
        }
        for diagnosis in sorted(total_by_diagnosis)
    }
    all_zero_mask_patients = sum(
        not has_positive for has_positive in patient_has_positive_mask.values()
    )
    summary = {
        "status": "completed",
        "experiment_code": "L0",
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "images": len(rows),
        "total_finding_objects": total_finding_objects,
        "zero_mask_images": zero_mask_images,
        "positive_mask_images": len(rows) - zero_mask_images,
        "patients": len(patient_has_positive_mask),
        "all_zero_mask_patients": all_zero_mask_patients,
        "any_positive_mask_patients": (
            len(patient_has_positive_mask) - all_zero_mask_patients
        ),
        "zero_mask_policy": config["zero_mask_policy"],
        "zero_mask_images_by_diagnosis": dict(sorted(zero_by_diagnosis.items())),
        "mask_presence_by_diagnosis": mask_presence_by_diagnosis,
        "zero_mask_interpretation": (
            "absence_of_selected_annotation_not_proof_of_no_medical_lesion"
        ),
        "included_categories": sorted(included_categories),
        "mask_size": [output_size, output_size],
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "contains_raw_paths": False,
        "category_names_exposed_to_classifier": False,
        "mask_presence_bias_requires_audit": True,
        "complete_rgb_preserved": True,
        "allow_2026": False,
    }
    summary_path = REPORT_DIR / "L0_manual_lesion_mask_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
