"""Deterministic and diagnosis-blinded annotation review queue selection."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


PUBLIC_REVIEW_FIELDS = (
    "review_case_key",
    "image_key",
    "target_category",
    "required_independent_reviews",
    "reviewer_1_presence_state",
    "reviewer_1_image_mode",
    "reviewer_1_annotation_scope",
    "reviewer_1_polygon_action",
    "reviewer_1_subtype",
    "reviewer_1_notes",
    "reviewer_2_presence_state",
    "reviewer_2_image_mode",
    "reviewer_2_annotation_scope",
    "reviewer_2_polygon_action",
    "reviewer_2_subtype",
    "reviewer_2_notes",
    "adjudicated_presence_state",
    "adjudicated_image_mode",
    "adjudicated_annotation_scope",
    "adjudicated_polygon_action",
    "adjudicated_subtype",
    "adjudication_notes",
)

FORBIDDEN_PUBLIC_FIELDS = {
    "diagnosis",
    "diagnosis_id",
    "outer_fold",
    "person_key",
    "legacy_annotation_state",
    "raw_image_path",
    "raw_annotation_path",
    "normalized_annotation_path",
}


def _stable_score(seed: int, *parts: str) -> str:
    value = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _choose_fold_bucket(
    pool: list[dict[str, Any]],
    count: int,
    seed: int,
    target: str,
    bucket: str,
    used_images: set[str],
) -> list[dict[str, Any]]:
    candidates = sorted(
        pool,
        key=lambda row: _stable_score(
            seed, target, bucket, str(row["image_key"])
        ),
    )
    diagnosis_counts = Counter()
    selected = []
    selected_people = set()
    while len(selected) < count:
        eligible = [
            row
            for row in candidates
            if str(row["image_key"]) not in used_images
            and str(row["person_key"]) not in selected_people
        ]
        if not eligible:
            raise ValueError(
                f"Insufficient unique review candidates for {target}/{bucket}"
            )
        minimum = min(diagnosis_counts[str(row["diagnosis"])] for row in eligible)
        row = next(
            item
            for item in eligible
            if diagnosis_counts[str(item["diagnosis"])] == minimum
        )
        selected.append(row)
        used_images.add(str(row["image_key"]))
        selected_people.add(str(row["person_key"]))
        diagnosis_counts[str(row["diagnosis"])] += 1
    return selected


def build_blinded_review_queue(
    image_rows: Iterable[Mapping[str, Any]],
    image_categories: Mapping[str, set[str]],
    targets: list[str],
    seed: int,
    per_fold_per_bucket: int,
    required_independent_reviews: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create public rows and a diagnosis-aware aggregate selection audit."""
    rows = [dict(row) for row in image_rows if int(row.get("include", 1)) == 1]
    used_images: set[str] = set()
    public_rows = []
    audit_rows = []

    for target in targets:
        for bucket in ("existing_positive", "legacy_unlabeled_candidate"):
            for fold in range(5):
                pool = [
                    row
                    for row in rows
                    if int(row["outer_fold"]) == fold
                    and (
                        target in image_categories.get(str(row["image_key"]), set())
                    )
                    == (bucket == "existing_positive")
                ]
                selected = _choose_fold_bucket(
                    pool,
                    per_fold_per_bucket,
                    seed,
                    target,
                    bucket,
                    used_images,
                )
                for row in selected:
                    image_key = str(row["image_key"])
                    case_key = "KNEE_REVIEW_" + _stable_score(
                        seed, target, image_key
                    )[:16].upper()
                    public_rows.append(
                        {
                            field: (
                                case_key
                                if field == "review_case_key"
                                else image_key
                                if field == "image_key"
                                else target
                                if field == "target_category"
                                else required_independent_reviews
                                if field == "required_independent_reviews"
                                else ""
                            )
                            for field in PUBLIC_REVIEW_FIELDS
                        }
                    )
                    audit_rows.append(
                        {
                            "target": target,
                            "bucket": bucket,
                            "fold": fold,
                            "diagnosis": str(row["diagnosis"]),
                            "person_key": str(row["person_key"]),
                        }
                    )

    if len({row["image_key"] for row in public_rows}) != len(public_rows):
        raise ValueError("Public review queue must contain unique images")
    if set(public_rows[0]) & FORBIDDEN_PUBLIC_FIELDS:
        raise ValueError("Public review queue contains a forbidden field")

    target_bucket_counts = Counter(
        (row["target"], row["bucket"]) for row in audit_rows
    )
    target_fold_counts = Counter((row["target"], row["fold"]) for row in audit_rows)
    target_diagnosis_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit_rows:
        target_diagnosis_counts[row["target"]][row["diagnosis"]] += 1
    audit = {
        "contract": {
            "rows": len(public_rows),
            "unique_images": len({row["image_key"] for row in public_rows}),
            "unique_patients_internal": len({row["person_key"] for row in audit_rows}),
            "targets": targets,
            "per_fold_per_bucket": per_fold_per_bucket,
            "required_independent_reviews": required_independent_reviews,
            "diagnosis_visible_to_reviewer": False,
            "legacy_annotation_visible_to_reviewer": False,
            "legacy_unlabeled_is_negative": False,
        },
        "target_bucket_counts": [
            {"target": target, "bucket": bucket, "count": count}
            for (target, bucket), count in sorted(target_bucket_counts.items())
        ],
        "target_fold_counts": [
            {"target": target, "fold": fold, "count": count}
            for (target, fold), count in sorted(target_fold_counts.items())
        ],
        "target_diagnosis_counts_internal_aggregate": {
            target: dict(sorted(counts.items()))
            for target, counts in sorted(target_diagnosis_counts.items())
        },
        "privacy": {
            "public_fields": list(PUBLIC_REVIEW_FIELDS),
            "forbidden_fields_absent": True,
            "patient_keys_not_exported": True,
            "source_paths_not_exported": True,
        },
    }
    return public_rows, audit
