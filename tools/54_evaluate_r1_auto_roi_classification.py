"""Merge R1 folds and compare frozen E1 predictions under automatic versus manual ROI."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_auto_roi import (  # noqa: E402
    descriptive_distribution,
    read_csv_rows,
    require_sha256,
    sha256_file,
)
from src.research_oof import (  # noqa: E402
    PROBABILITY_COLUMNS,
    compare_oof_files,
    evaluate_oof_file,
    merge_oof_fold_files,
)


CONFIG_PATH = ROOT / "configs/research/r1_auto_roi_classification_sensitivity.yaml"


def probability_vector(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in PROBABILITY_COLUMNS], dtype=np.float64)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    fold_paths = [
        PATIENT_MULTIMODAL_REPORTS_DIR / "oof" / f"R1_fold{fold}.csv"
        for fold in config["evaluation"]["outer_folds"]
    ]
    for path in fold_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_path = PATIENT_MULTIMODAL_REPORTS_DIR / "oof" / "R1_oof.csv"
    merge_oof_fold_files(fold_paths, candidate_path)
    baseline_path = (ROOT / config["base_experiment"]["manual_oof_path"]).resolve()
    full_image_path = (ROOT / config["full_image_control"]["oof_path"]).resolve()
    frozen_inputs = (
        (
            baseline_path,
            config["base_experiment"]["manual_oof_sha256"],
            "manual ROI OOF",
        ),
        (
            full_image_path,
            config["full_image_control"]["oof_sha256"],
            "full-image OOF",
        ),
    )
    for path, expected_hash, label in frozen_inputs:
        require_sha256(path, expected_hash, label)
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    baseline_evaluation = evaluate_oof_file(
        baseline_path,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    candidate_evaluation = evaluate_oof_file(
        candidate_path,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    paired = compare_oof_files(
        baseline_path,
        candidate_path,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    full_image_evaluation = evaluate_oof_file(
        full_image_path,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    paired_vs_full_image = compare_oof_files(
        full_image_path,
        candidate_path,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )

    baseline_per_class = {
        int(row["class_id"]): row for row in baseline_evaluation["metrics"]["per_class"]
    }
    candidate_per_class = {
        int(row["class_id"]): row for row in candidate_evaluation["metrics"]["per_class"]
    }
    per_class_delta = []
    for class_id, baseline_row in sorted(baseline_per_class.items()):
        candidate_row = candidate_per_class[class_id]
        per_class_delta.append(
            {
                "class_id": class_id,
                "class_name": baseline_row["class_name"],
                "sensitivity_delta": candidate_row["sensitivity"] - baseline_row["sensitivity"],
                "f1_delta": candidate_row["f1"] - baseline_row["f1"],
            }
        )

    image_manifest = {
        row["image_key"]: row
        for row in read_csv_rows(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv")
        if row["include"] == "1"
    }
    patient_ious: dict[str, list[float]] = defaultdict(list)
    fallback_by_patient: dict[str, int] = defaultdict(int)
    for fold in config["evaluation"]["outer_folds"]:
        r0_path = ROOT / config["candidate_input"]["pattern"].format(fold=fold)
        for row in read_csv_rows(r0_path):
            person_key = image_manifest[row["image_key"]]["person_key"]
            if row["status"] == "detected":
                patient_ious[person_key].append(float(row["reference_iou_descriptive"]))
            else:
                fallback_by_patient[person_key] += 1

    baseline_rows = {row["person_key"]: row for row in read_csv_rows(baseline_path)}
    candidate_rows = {row["person_key"]: row for row in read_csv_rows(candidate_path)}
    changed_rows = []
    stable_iou = []
    changed_iou = []
    for person_key in sorted(candidate_rows):
        baseline_row = baseline_rows[person_key]
        candidate_row = candidate_rows[person_key]
        baseline_probabilities = probability_vector(baseline_row)
        candidate_probabilities = probability_vector(candidate_row)
        baseline_prediction = int(np.argmax(baseline_probabilities))
        candidate_prediction = int(np.argmax(candidate_probabilities))
        reference = int(candidate_row["reference_id"])
        ious = patient_ious.get(person_key, [])
        mean_iou = float(np.mean(ious)) if ious else None
        minimum_iou = float(np.min(ious)) if ious else None
        changed = baseline_prediction != candidate_prediction
        if mean_iou is not None:
            (changed_iou if changed else stable_iou).append(mean_iou)
        if changed:
            changed_rows.append(
                {
                    "person_key": person_key,
                    "outer_fold": candidate_row["outer_fold"],
                    "reference_id": reference,
                    "manual_roi_prediction_id": baseline_prediction,
                    "auto_roi_prediction_id": candidate_prediction,
                    "manual_roi_correct": int(baseline_prediction == reference),
                    "auto_roi_correct": int(candidate_prediction == reference),
                    "probability_l1": float(np.abs(candidate_probabilities - baseline_probabilities).sum()),
                    "mean_reference_iou_descriptive": mean_iou,
                    "minimum_reference_iou_descriptive": minimum_iou,
                    "full_image_fallback_count": fallback_by_patient.get(person_key, 0),
                }
            )

    changed_path = PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_changed_patient_predictions.csv"
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(changed_rows[0]) if changed_rows else ["person_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed_rows)

    summary = {
        "experiment_code": "R1",
        "status": "completed_no_hard_gate",
        "classifier_retrained": False,
        "patients": int(config["evaluation"]["patients"]),
        "baseline_manual_roi": baseline_evaluation,
        "candidate_auto_roi": candidate_evaluation,
        "paired_auto_minus_manual": paired,
        "full_image_risk_control": full_image_evaluation,
        "paired_auto_minus_full_image_contextual": paired_vs_full_image,
        "comparison_interpretation": {
            "primary": "Auto versus manual ROI uses the same frozen E1 checkpoints and isolates inference-time ROI source.",
            "contextual": "Auto ROI versus E0 compares practical routes but also differs in classifier training input.",
        },
        "per_class_delta": per_class_delta,
        "changed_top1_patients": len(changed_rows),
        "manual_correct_auto_wrong": sum(
            row["manual_roi_correct"] == 1 and row["auto_roi_correct"] == 0
            for row in changed_rows
        ),
        "manual_wrong_auto_correct": sum(
            row["manual_roi_correct"] == 0 and row["auto_roi_correct"] == 1
            for row in changed_rows
        ),
        "mean_patient_iou_when_top1_stable": descriptive_distribution(stable_iou),
        "mean_patient_iou_when_top1_changed": descriptive_distribution(changed_iou),
        "changed_patient_audit": {
            "path": changed_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(changed_path),
        },
        "geometry_gate": None,
        "decision_basis": config["evaluation"]["decision_basis"],
        "allow_2026_labels": False,
    }
    output_path = PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_classification_sensitivity.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
