"""Export privacy-safe outer-test ROI predictions from a frozen R0 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_auto_roi import ROIBox, load_r0_config, read_csv_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/r0_auto_ultrasound_view_roi.yaml"),
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_r0_config((ROOT / args.config).resolve())
    dataset_dir = (ROOT / config["data"]["output_dir"] / f"fold_{args.fold}").resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    manifest_rows = {
        row["image_key"]: row
        for row in read_csv_rows(dataset_dir / "manifest.csv")
        if row["split"] == "test"
    }
    test_dir = dataset_dir / "images/test"
    if len(manifest_rows) != len(list(test_dir.glob("*"))):
        raise ValueError("R0 test manifest and image directory counts differ")

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    predictions: list[dict[str, object]] = []
    ious: list[float] = []
    missing = 0
    results = model.predict(
        source=str(test_dir),
        imgsz=int(config["model"]["image_size"]),
        conf=float(args.confidence),
        device=config["runtime"]["device"],
        stream=True,
        verbose=False,
    )
    for result in results:
        image_key = Path(result.path).stem
        reference_row = manifest_rows[image_key]
        reference = ROIBox.from_mapping(reference_row)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            missing += 1
            predictions.append(
                {
                    "image_key": image_key,
                    "outer_fold": args.fold,
                    "confidence": "",
                    "auto_roi_x1": "",
                    "auto_roi_y1": "",
                    "auto_roi_x2": "",
                    "auto_roi_y2": "",
                    "reference_iou_descriptive": "",
                    "status": "no_detection",
                }
            )
            continue
        best_index = int(boxes.conf.argmax().item())
        coords = [float(value) for value in boxes.xyxy[best_index].tolist()]
        predicted = ROIBox(*coords)
        iou = predicted.iou(reference)
        ious.append(iou)
        predictions.append(
            {
                "image_key": image_key,
                "outer_fold": args.fold,
                "confidence": float(boxes.conf[best_index].item()),
                "auto_roi_x1": predicted.x1,
                "auto_roi_y1": predicted.y1,
                "auto_roi_x2": predicted.x2,
                "auto_roi_y2": predicted.y2,
                "reference_iou_descriptive": iou,
                "status": "detected",
            }
        )

    if len(predictions) != len(manifest_rows):
        raise ValueError("R0 inference did not return exactly one result per outer-test image")
    predictions.sort(key=lambda row: str(row["image_key"]))
    report_dir = (
        ROOT
        / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/auto_roi"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"R0_fold_{args.fold}_oof_roi.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "fold": args.fold,
        "images": len(predictions),
        "detections": len(ious),
        "no_detection": missing,
        "mean_reference_iou_descriptive": float(np.mean(ious)) if ious else None,
        "median_reference_iou_descriptive": float(np.median(ious)) if ious else None,
        "geometry_gate": None,
        "primary_decision_basis": "downstream_patient_classification",
        "contains_raw_paths": False,
        "checkpoint": checkpoint.name,
    }
    summary_path = report_dir / f"R0_fold_{args.fold}_oof_roi_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

