"""Merge R1 folds and compare frozen E1 predictions under automatic versus manual ROI."""

from __future__ import annotations

import csv
import json
import random
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
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/r1_auto_roi_classification_sensitivity.yaml"


def probability_vector(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[column]) for column in PROBABILITY_COLUMNS], dtype=np.float64)


CASE_REVIEW_REASON_ORDER = (
    "manual_correct_auto_wrong",
    "full_image_fallback",
    "largest_probability_shift",
    "lowest_mean_iou_descriptive",
    "stable_top1_random_control",
)


def select_case_review_patients(
    patient_rows: list[dict[str, object]], review_config: dict[str, object]
) -> dict[str, tuple[str, ...]]:
    """Apply the preregistered, deterministic R1 case-review sampling rules."""

    reasons: dict[str, set[str]] = defaultdict(set)
    if review_config["include_all_manual_correct_auto_wrong"]:
        for row in patient_rows:
            if row["manual_roi_correct"] == 1 and row["auto_roi_correct"] == 0:
                reasons[str(row["person_key"])].add("manual_correct_auto_wrong")
    if review_config["include_all_full_image_fallback"]:
        for row in patient_rows:
            if int(row["full_image_fallback_count"]) > 0:
                reasons[str(row["person_key"])].add("full_image_fallback")

    probability_shift = sorted(
        patient_rows,
        key=lambda row: (-float(row["probability_l1"]), str(row["person_key"])),
    )[: int(review_config["top_probability_shift_patients"])]
    for row in probability_shift:
        reasons[str(row["person_key"])].add("largest_probability_shift")

    with_geometry = [
        row for row in patient_rows if row["mean_reference_iou_descriptive"] is not None
    ]
    lowest_iou = sorted(
        with_geometry,
        key=lambda row: (
            float(row["mean_reference_iou_descriptive"]),
            str(row["person_key"]),
        ),
    )[: int(review_config["lowest_mean_iou_patients_descriptive"])]
    for row in lowest_iou:
        reasons[str(row["person_key"])].add("lowest_mean_iou_descriptive")

    stable = sorted(
        (row for row in patient_rows if row["top1_changed"] == 0),
        key=lambda row: str(row["person_key"]),
    )
    sample_count = min(int(review_config["stable_top1_random_patients"]), len(stable))
    sampled = random.Random(int(review_config["random_seed"])).sample(
        stable, sample_count
    )
    for row in sampled:
        reasons[str(row["person_key"])].add("stable_top1_random_control")

    return {
        person_key: tuple(
            reason for reason in CASE_REVIEW_REASON_ORDER if reason in observed
        )
        for person_key, observed in reasons.items()
    }


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
    r0_by_image: dict[str, dict[str, str]] = {}
    for fold in config["evaluation"]["outer_folds"]:
        r0_path = ROOT / config["candidate_input"]["pattern"].format(fold=fold)
        for row in read_csv_rows(r0_path):
            if row["image_key"] in r0_by_image:
                raise ValueError("R0 OOF contains a duplicate image key across folds")
            r0_by_image[row["image_key"]] = row
            person_key = image_manifest[row["image_key"]]["person_key"]
            if row["status"] == "detected":
                patient_ious[person_key].append(float(row["reference_iou_descriptive"]))
            else:
                fallback_by_patient[person_key] += 1

    baseline_rows = {row["person_key"]: row for row in read_csv_rows(baseline_path)}
    candidate_rows = {row["person_key"]: row for row in read_csv_rows(candidate_path)}
    changed_rows = []
    patient_rows = []
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
        audit_row = {
            "person_key": person_key,
            "outer_fold": candidate_row["outer_fold"],
            "reference_id": reference,
            "reference_class": DIAGNOSIS_CLASSES[reference],
            "manual_roi_prediction_id": baseline_prediction,
            "manual_roi_prediction_class": DIAGNOSIS_CLASSES[baseline_prediction],
            "auto_roi_prediction_id": candidate_prediction,
            "auto_roi_prediction_class": DIAGNOSIS_CLASSES[candidate_prediction],
            "manual_roi_correct": int(baseline_prediction == reference),
            "auto_roi_correct": int(candidate_prediction == reference),
            "top1_changed": int(changed),
            "probability_l1": float(
                np.abs(candidate_probabilities - baseline_probabilities).sum()
            ),
            "reference_probability_delta_auto_minus_manual": float(
                candidate_probabilities[reference] - baseline_probabilities[reference]
            ),
            "mean_reference_iou_descriptive": mean_iou,
            "minimum_reference_iou_descriptive": minimum_iou,
            "full_image_fallback_count": fallback_by_patient.get(person_key, 0),
        }
        patient_rows.append(audit_row)
        if changed:
            changed_rows.append(audit_row)

    changed_path = PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_changed_patient_predictions.csv"
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(changed_rows[0]) if changed_rows else ["person_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed_rows)

    review_reasons = select_case_review_patients(patient_rows, config["case_review"])
    review_priority = {reason: index for index, reason in enumerate(CASE_REVIEW_REASON_ORDER)}
    patient_by_key = {str(row["person_key"]): row for row in patient_rows}
    ordered_review_keys = sorted(
        review_reasons,
        key=lambda person_key: (
            min(review_priority[reason] for reason in review_reasons[person_key]),
            -float(patient_by_key[person_key]["probability_l1"]),
            person_key,
        ),
    )
    review_queue = []
    for rank, person_key in enumerate(ordered_review_keys, start=1):
        review_queue.append(
            {
                "review_rank": rank,
                "review_reasons": "|".join(review_reasons[person_key]),
                **patient_by_key[person_key],
            }
        )
    review_queue_path = (
        PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_case_review_queue.csv"
    )
    with review_queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_queue[0]))
        writer.writeheader()
        writer.writerows(review_queue)

    review_rank_by_person = {
        str(row["person_key"]): int(row["review_rank"]) for row in review_queue
    }
    review_image_rows = []
    for image_key, manifest_row in sorted(image_manifest.items()):
        person_key = manifest_row["person_key"]
        if person_key not in review_reasons:
            continue
        r0_row = r0_by_image[image_key]
        review_image_rows.append(
            {
                "review_rank": review_rank_by_person[person_key],
                "review_reasons": "|".join(review_reasons[person_key]),
                "person_key": person_key,
                "image_key": image_key,
                "outer_fold": r0_row["outer_fold"],
                "r0_status": r0_row["status"],
                "r0_confidence": r0_row["confidence"],
                **{
                    f"auto_roi_{axis}": r0_row[f"auto_roi_{axis}"]
                    for axis in ("x1", "y1", "x2", "y2")
                },
                **{
                    f"manual_roi_{axis}": manifest_row[f"roi_{axis}"]
                    for axis in ("x1", "y1", "x2", "y2")
                },
                "reference_iou_descriptive": r0_row["reference_iou_descriptive"],
            }
        )
    review_images_path = (
        PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_case_review_images.csv"
    )
    with review_images_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_image_rows[0]))
        writer.writeheader()
        writer.writerows(review_image_rows)

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
        "case_review": {
            "preregistered_sampling": config["case_review"],
            "patients": len(review_queue),
            "images": len(review_image_rows),
            "reason_counts": {
                reason: sum(reason in observed for observed in review_reasons.values())
                for reason in CASE_REVIEW_REASON_ORDER
            },
            "patient_queue_path": review_queue_path.relative_to(ROOT).as_posix(),
            "patient_queue_sha256": sha256_file(review_queue_path),
            "image_queue_path": review_images_path.relative_to(ROOT).as_posix(),
            "image_queue_sha256": sha256_file(review_images_path),
            "contains_raw_paths": False,
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
