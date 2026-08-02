"""Aggregate, privacy-safe audits for polygon supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon

from src.label_mapping import get_disease_from_label
from src.research_metrics import (
    bootstrap_macro_auc_ci,
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
)
from src.research_schema import DIAGNOSIS_CLASSES


def _polygon(points: object):
    if not isinstance(points, list) or len(points) < 3:
        return None, False
    try:
        shape = Polygon([(float(point[0]), float(point[1])) for point in points])
    except (TypeError, ValueError, IndexError):
        return None, False
    originally_valid = bool(shape.is_valid and not shape.is_empty and shape.area > 0)
    if not originally_valid:
        shape = make_valid(shape)
    if shape.is_empty or shape.area <= 0:
        return None, originally_valid
    return shape, originally_valid


def _configured_support_tier(
    patient_count: int,
    fold_counts: list[int],
    thresholds: Mapping[str, int],
) -> str:
    if (
        patient_count >= int(thresholds["robust_min_patients"])
        and min(fold_counts) >= int(thresholds["robust_min_patients_per_fold"])
    ):
        return "robust_multifold"
    if (
        patient_count >= int(thresholds["limited_min_patients"])
        and min(fold_counts) >= int(thresholds["limited_min_patients_per_fold"])
    ):
        return "limited_multifold"
    return "insufficient_multifold"


def audit_annotation_records(
    image_rows: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    categories: list[str],
    category_roles: Mapping[str, str],
    support_thresholds: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Audit linked annotations without returning source paths or identities."""
    category_set = set(categories)
    if set(category_roles) != category_set:
        raise ValueError("Every category must have exactly one configured role")
    if set(category_roles.values()) - {"anatomy", "finding", "ambiguous"}:
        raise ValueError("Unsupported annotation role")
    support_thresholds = support_thresholds or {
        "robust_min_patients": 50,
        "robust_min_patients_per_fold": 5,
        "limited_min_patients": 20,
        "limited_min_patients_per_fold": 2,
    }

    rows = [dict(row) for row in image_rows if int(row.get("include", 1)) == 1]
    patient_diagnosis: dict[str, str] = {}
    patient_labels: dict[str, set[str]] = defaultdict(set)
    category_objects = Counter()
    category_images: dict[str, set[str]] = defaultdict(set)
    category_patients: dict[str, set[str]] = defaultdict(set)
    category_fold_patients: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    category_diagnosis_patients: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    category_area_fractions: dict[str, list[float]] = defaultdict(list)
    cooccurrence_images = Counter()
    overlap_pairs = Counter()
    overlap_pair_images: dict[tuple[str, str], set[str]] = defaultdict(set)
    counters = Counter()
    linked_annotation_keys = set()

    for row in rows:
        image_key = str(row["image_key"])
        person_key = str(row["person_key"])
        diagnosis = str(row["diagnosis"])
        outer_fold = int(row["outer_fold"])
        patient_diagnosis[person_key] = diagnosis
        annotation = annotations.get(image_key)
        if annotation is None:
            counters["images_without_annotation_json"] += 1
            continue
        linked_annotation_keys.add(image_key)
        counters["images_with_annotation_json"] += 1
        objects = annotation.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("Annotation objects must be a list")
        if not objects:
            counters["annotated_images_with_zero_objects"] += 1

        info = annotation.get("info", {})
        width = float(info.get("width") or row.get("width") or 0)
        height = float(info.get("height") or row.get("height") or 0)
        roi = annotation.get("ultrasound_rect") or {}
        roi_bounds = (
            float(roi.get("x1", 0)),
            float(roi.get("y1", 0)),
            float(roi.get("x2", width)),
            float(roi.get("y2", height)),
        )
        roi_area = max(0.0, roi_bounds[2] - roi_bounds[0]) * max(
            0.0, roi_bounds[3] - roi_bounds[1]
        )
        image_categories = set()
        image_shapes = []
        for item in objects:
            counters["objects"] += 1
            category = str(item.get("category", "")).strip()
            if category not in category_set:
                counters["unknown_category_objects"] += 1
                continue
            if get_disease_from_label(category) is not None:
                counters["residual_disease_prefix_objects"] += 1
            category_objects[category] += 1
            category_images[category].add(image_key)
            category_patients[category].add(person_key)
            category_fold_patients[category][outer_fold].add(person_key)
            category_diagnosis_patients[category][diagnosis].add(person_key)
            patient_labels[person_key].add(category)
            image_categories.add(category)

            shape, originally_valid = _polygon(item.get("segmentation"))
            if shape is None:
                counters["unusable_polygon_objects"] += 1
                continue
            if not originally_valid:
                counters["repaired_invalid_polygon_objects"] += 1
            counters["usable_polygon_objects"] += 1
            min_x, min_y, max_x, max_y = shape.bounds
            if min_x < 0 or min_y < 0 or max_x > width or max_y > height:
                counters["objects_outside_image_bounds"] += 1
            if (
                min_x < roi_bounds[0]
                or min_y < roi_bounds[1]
                or max_x > roi_bounds[2]
                or max_y > roi_bounds[3]
            ):
                counters["objects_not_fully_inside_roi"] += 1
            if roi_area > 0:
                category_area_fractions[category].append(float(shape.area / roi_area))

            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                if max(abs(float(a) - float(b)) for a, b in zip(bbox, shape.bounds)) > 2.0:
                    counters["bbox_polygon_mismatch_objects"] += 1
            else:
                counters["missing_or_invalid_bbox_objects"] += 1
            recorded_area = item.get("area")
            if recorded_area is None or float(recorded_area) <= 0:
                counters["missing_or_invalid_area_objects"] += 1
            elif abs(float(recorded_area) - shape.area) / max(shape.area, 1.0) > 0.05:
                counters["area_polygon_mismatch_objects"] += 1
            image_shapes.append((category, shape))

        for left, right in combinations(sorted(image_categories), 2):
            cooccurrence_images[(left, right)] += 1
        for (left_category, left_shape), (right_category, right_shape) in combinations(
            image_shapes, 2
        ):
            if not left_shape.intersects(right_shape):
                continue
            intersection_area = float(left_shape.intersection(right_shape).area)
            if intersection_area <= 1e-6:
                continue
            pair = tuple(sorted((left_category, right_category)))
            overlap_pairs[pair] += 1
            overlap_pair_images[pair].add(image_key)

    category_rows = []
    for category in categories:
        fold_counts = [len(category_fold_patients[category][fold]) for fold in range(5)]
        diagnosis_counts = {
            diagnosis: len(category_diagnosis_patients[category][diagnosis])
            for diagnosis in DIAGNOSIS_CLASSES
        }
        patient_count = len(category_patients[category])
        top_diagnosis, top_count = max(
            diagnosis_counts.items(), key=lambda item: (item[1], item[0])
        )
        fractions = category_area_fractions[category]
        category_rows.append(
            {
                "category": category,
                "role": category_roles[category],
                "objects": category_objects[category],
                "images": len(category_images[category]),
                "patients": patient_count,
                "fold_patient_counts": fold_counts,
                "diagnosis_patient_counts": diagnosis_counts,
                "top_diagnosis": top_diagnosis,
                "top_diagnosis_patient_share": (
                    top_count / patient_count if patient_count else 0.0
                ),
                "support_tier": _configured_support_tier(
                    patient_count, fold_counts, support_thresholds
                ),
                "median_polygon_to_roi_area": (
                    float(np.median(fractions)) if fractions else None
                ),
            }
        )

    unlinked_annotation_count = len(set(annotations) - linked_annotation_keys)
    report = {
        "contract": {
            "images": len(rows),
            "patients": len(patient_diagnosis),
            "categories": len(categories),
            "folds": [0, 1, 2, 3, 4],
            "support_thresholds": dict(support_thresholds),
            "privacy": "deidentified_aggregate_output",
        },
        "coverage": {
            **dict(counters),
            "unlinked_annotation_records": unlinked_annotation_count,
            "annotation_json_coverage": (
                counters["images_with_annotation_json"] / len(rows) if rows else 0.0
            ),
            "object_annotation_image_coverage": (
                (
                    counters["images_with_annotation_json"]
                    - counters["annotated_images_with_zero_objects"]
                )
                / len(rows)
                if rows
                else 0.0
            ),
            "absence_as_true_negative": "not_established_without_exhaustive_protocol",
        },
        "categories": category_rows,
        "support_summary": dict(Counter(row["support_tier"] for row in category_rows)),
        "diagnosis_concentration": {
            "categories_with_top_share_ge_0_80": sum(
                row["top_diagnosis_patient_share"] >= 0.80 for row in category_rows
            ),
            "categories_with_top_share_ge_0_95": sum(
                row["top_diagnosis_patient_share"] >= 0.95 for row in category_rows
            ),
        },
        "top_image_cooccurrences": [
            {"categories": list(pair), "images": count}
            for pair, count in cooccurrence_images.most_common(30)
        ],
        "polygon_overlaps": {
            "overlapping_object_pairs": sum(overlap_pairs.values()),
            "same_category_pairs": sum(
                count for pair, count in overlap_pairs.items() if pair[0] == pair[1]
            ),
            "cross_category_pairs": sum(
                count for pair, count in overlap_pairs.items() if pair[0] != pair[1]
            ),
            "top_pairs": [
                {
                    "categories": list(pair),
                    "object_pairs": count,
                    "images": len(overlap_pair_images[pair]),
                }
                for pair, count in overlap_pairs.most_common(30)
            ],
            "single_multiclass_mask_safe": sum(
                count for pair, count in overlap_pairs.items() if pair[0] != pair[1]
            )
            == 0,
        },
    }
    return report, patient_labels


def evaluate_manual_presence_proxy(
    patient_rows: list[Mapping[str, Any]],
    patient_labels: Mapping[str, set[str]],
    categories: list[str],
    category_roles: Mapping[str, str],
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260724,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Cross-fit diagnosis from manual label presence as a workflow-bias audit."""
    from sklearn.linear_model import LogisticRegression

    rows = sorted(patient_rows, key=lambda item: str(item["person_key"]))
    y = np.asarray([int(row["diagnosis_id"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    feature_groups = {
        "all": categories,
        "anatomy": [c for c in categories if category_roles[c] == "anatomy"],
        "finding": [c for c in categories if category_roles[c] == "finding"],
    }
    result = {}
    all_oof = {}
    model_weights = {}
    for group, group_categories in feature_groups.items():
        matrix = np.asarray(
            [
                [float(category in patient_labels.get(str(row["person_key"]), set())) for category in group_categories]
                for row in rows
            ],
            dtype=np.float64,
        )
        probabilities = np.zeros((len(rows), len(DIAGNOSIS_CLASSES)), dtype=np.float64)
        fold_macro_f1 = []
        group_weights = []
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=20260724 + fold,
            )
            model.fit(matrix[train], y[train])
            probabilities[test] = model.predict_proba(matrix[test])
            group_weights.append(
                {
                    "fold": fold,
                    "classes": [int(value) for value in model.classes_],
                    "feature_categories": group_categories,
                    "coef": model.coef_.tolist(),
                    "intercept": model.intercept_.tolist(),
                    "n_iter": [int(value) for value in model.n_iter_],
                }
            )
            fold_macro_f1.append(
                compute_patient_metrics(
                    y[test], probabilities[test], DIAGNOSIS_CLASSES
                )["macro_f1"]
            )
        metrics = compute_patient_metrics(y, probabilities, DIAGNOSIS_CLASSES)
        f1_interval = bootstrap_macro_f1_ci(
            y, probabilities, bootstrap_samples, bootstrap_seed
        )
        auc_interval = bootstrap_macro_auc_ci(
            y, probabilities, bootstrap_samples, bootstrap_seed
        )
        result[group] = {
            "feature_count": len(group_categories),
            "metrics": metrics,
            "fold_macro_f1": fold_macro_f1,
            "confidence_intervals": {
                "method": "patient_stratified_bootstrap_percentile",
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "macro_f1_ci95": [f1_interval[0], f1_interval[2]],
                "macro_auc_ci95": [auc_interval[0], auc_interval[2]],
            },
        }
        all_oof[group] = probabilities
        model_weights[group] = group_weights

    output_rows = []
    for index, row in enumerate(rows):
        output = {
            "person_key": str(row["person_key"]),
            "outer_fold": int(row["outer_fold"]),
            "reference_class": str(row["diagnosis"]),
            "reference_id": int(row["diagnosis_id"]),
        }
        for group, probabilities in all_oof.items():
            for class_id in range(len(DIAGNOSIS_CLASSES)):
                output[f"{group}_prob_{class_id}"] = float(probabilities[index, class_id])
        output_rows.append(output)
    return result, output_rows, model_weights
