"""Merge and describe fivefold R0 OOF boxes without applying a geometry gate."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_auto_roi import (  # noqa: E402
    descriptive_distribution,
    load_r0_config,
    read_csv_rows,
    sha256_file,
)


CONFIG_PATH = ROOT / "configs/research/r0_auto_ultrasound_view_roi.yaml"
REPORT_DIR = (
    ROOT
    / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/auto_roi"
)


def main() -> int:
    config = load_r0_config(CONFIG_PATH)
    registry_dir = (ROOT / config["data"]["registry_dir"]).resolve()
    image_rows = {
        row["image_key"]: row
        for row in read_csv_rows(registry_dir / config["data"]["images_csv"])
        if row["include"] == "1"
    }
    outer_fold_by_person = {
        row["person_key"]: int(row["outer_fold"])
        for row in read_csv_rows(registry_dir / "folds_outer.csv")
    }

    predictions: list[dict[str, str]] = []
    source_files: list[dict[str, str]] = []
    for fold in config["evaluation"]["outer_folds"]:
        path = REPORT_DIR / f"R0_fold_{fold}_oof_roi.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing formal R0 OOF export: {path}")
        rows = read_csv_rows(path)
        predictions.extend(rows)
        source_files.append({"fold": str(fold), "path": path.name, "sha256": sha256_file(path)})

    expected_images = int(config["data"]["expected_images"])
    if len(predictions) != expected_images:
        raise ValueError(f"expected {expected_images} OOF rows, found {len(predictions)}")
    keys = [row["image_key"] for row in predictions]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate image_key in merged R0 OOF predictions")
    if set(keys) != set(image_rows):
        raise ValueError("R0 OOF image keys do not exactly match the development manifest")

    detected_rows: list[dict[str, object]] = []
    no_detection_rows: list[dict[str, object]] = []
    iou_by_fold: dict[int, list[float]] = defaultdict(list)
    iou_by_diagnosis: dict[str, list[float]] = defaultdict(list)
    edge_errors: dict[str, list[float]] = defaultdict(list)

    for prediction in predictions:
        manifest = image_rows[prediction["image_key"]]
        expected_fold = outer_fold_by_person[manifest["person_key"]]
        actual_fold = int(prediction["outer_fold"])
        if actual_fold != expected_fold:
            raise ValueError(
                f"OOF fold mismatch for {prediction['image_key']}: {actual_fold} != {expected_fold}"
            )
        if prediction["status"] == "no_detection":
            no_detection_rows.append(
                {
                    "image_key": prediction["image_key"],
                    "person_key": manifest["person_key"],
                    "outer_fold": actual_fold,
                    "diagnosis": manifest["diagnosis"],
                    "status": "no_detection",
                }
            )
            continue
        if prediction["status"] != "detected":
            raise ValueError(f"unsupported prediction status: {prediction['status']}")

        iou = float(prediction["reference_iou_descriptive"])
        width, height = float(manifest["width"]), float(manifest["height"])
        for axis, scale in (("x1", width), ("x2", width), ("y1", height), ("y2", height)):
            auto = float(prediction[f"auto_roi_{axis}"])
            manual = float(manifest[f"roi_{axis}"])
            edge_errors[axis].append(abs(auto - manual) / scale)
        iou_by_fold[actual_fold].append(iou)
        iou_by_diagnosis[manifest["diagnosis"]].append(iou)
        detected_rows.append(
            {
                "image_key": prediction["image_key"],
                "person_key": manifest["person_key"],
                "outer_fold": actual_fold,
                "diagnosis": manifest["diagnosis"],
                "confidence": float(prediction["confidence"]),
                "reference_iou_descriptive": iou,
                "status": "detected",
            }
        )

    detected_rows.sort(key=lambda row: float(row["reference_iou_descriptive"]))
    audit_rows = no_detection_rows + detected_rows[:100]
    audit_path = REPORT_DIR / "R0_oof_low_agreement_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image_key",
            "person_key",
            "outer_fold",
            "diagnosis",
            "confidence",
            "reference_iou_descriptive",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "experiment_code": "R0",
        "status": "completed_geometry_descriptive_only",
        "images": len(predictions),
        "detections": len(detected_rows),
        "no_detection": len(no_detection_rows),
        "geometry_gate": None,
        "primary_decision_basis": "downstream_patient_classification",
        "reference_iou": descriptive_distribution(
            float(row["reference_iou_descriptive"]) for row in detected_rows
        ),
        "normalized_absolute_edge_error": {
            axis: descriptive_distribution(values) for axis, values in sorted(edge_errors.items())
        },
        "reference_iou_by_fold": {
            str(fold): descriptive_distribution(values) for fold, values in sorted(iou_by_fold.items())
        },
        "reference_iou_by_diagnosis": {
            diagnosis: descriptive_distribution(values)
            for diagnosis, values in sorted(iou_by_diagnosis.items())
        },
        "source_oof_files": source_files,
        "low_agreement_audit": {
            "path": audit_path.name,
            "sha256": sha256_file(audit_path),
            "rows": len(audit_rows),
        },
        "contains_raw_paths": False,
        "allow_2026_labels": False,
    }
    output_path = REPORT_DIR / "R0_oof_geometry_descriptive.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

