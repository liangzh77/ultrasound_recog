"""Deterministic patient selection for the frozen L0 lesion-mask case review."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


PROBABILITY_COLUMNS = (
    "prob_normal",
    "prob_ra",
    "prob_ga",
    "prob_spa",
    "prob_oa",
    "prob_injury",
)


def build_case_review_queue(
    l0_rows: Iterable[Mapping[str, str]],
    e1_rows: Iterable[Mapping[str, str]],
    positive_images_by_person: Mapping[str, int],
    *,
    per_class_changed: int = 2,
    per_class_drift: int = 2,
    per_class_stable_mask_group: int = 1,
) -> list[dict[str, object]]:
    """Select representative patients without changing any trained model or result.

    The queue covers L0 corrections, L0-added errors, largest probability drift and
    stable-correct controls from positive- and all-zero-mask patient groups. Ties are
    resolved by deidentified patient key so repeated runs are byte-stable.
    """

    l0 = {row["person_key"]: dict(row) for row in l0_rows}
    e1 = {row["person_key"]: dict(row) for row in e1_rows}
    if not l0 or set(l0) != set(e1):
        raise ValueError("L0 and E1 OOF rows must cover the same non-empty patients")
    if set(l0) != set(positive_images_by_person):
        raise ValueError("Mask presence must cover exactly the OOF patients")

    candidates: list[dict[str, object]] = []
    for person_key in sorted(l0):
        l0_row = l0[person_key]
        e1_row = e1[person_key]
        for field in ("outer_fold", "reference_id", "reference_class"):
            if l0_row[field] != e1_row[field]:
                raise ValueError(f"L0/E1 mismatch for {person_key}: {field}")
        reference = l0_row["reference_class"]
        l0_correct = l0_row["top1"] == reference
        e1_correct = e1_row["top1"] == reference
        drift = sum(
            abs(float(l0_row[column]) - float(e1_row[column]))
            for column in PROBABILITY_COLUMNS
        )
        candidates.append(
            {
                "person_key": person_key,
                "outer_fold": int(l0_row["outer_fold"]),
                "reference_class": reference,
                "reference_id": int(l0_row["reference_id"]),
                "image_count": int(l0_row["image_count"]),
                "positive_mask_image_count": int(positive_images_by_person[person_key]),
                "mask_group": (
                    "any_positive"
                    if int(positive_images_by_person[person_key]) > 0
                    else "all_zero"
                ),
                "l0_top1": l0_row["top1"],
                "e1_top1": e1_row["top1"],
                "l0_correct": l0_correct,
                "e1_correct": e1_correct,
                "probability_l1_drift": drift,
            }
        )

    selected: dict[str, set[str]] = defaultdict(set)
    classes = sorted({str(row["reference_class"]) for row in candidates})

    def take(reason: str, rows: list[dict[str, object]], count: int) -> None:
        ordered = sorted(
            rows,
            key=lambda row: (-float(row["probability_l1_drift"]), str(row["person_key"])),
        )
        for row in ordered[:count]:
            selected[str(row["person_key"])].add(reason)

    for reference_class in classes:
        group = [row for row in candidates if row["reference_class"] == reference_class]
        take(
            "l0_corrected_e1_error",
            [row for row in group if row["l0_correct"] and not row["e1_correct"]],
            per_class_changed,
        )
        take(
            "l0_added_error",
            [row for row in group if row["e1_correct"] and not row["l0_correct"]],
            per_class_changed,
        )
        take("largest_probability_drift", group, per_class_drift)
        take(
            "stable_correct_positive_mask_control",
            [
                row
                for row in group
                if row["l0_correct"]
                and row["e1_correct"]
                and row["mask_group"] == "any_positive"
            ],
            per_class_stable_mask_group,
        )
        take(
            "stable_correct_all_zero_mask_control",
            [
                row
                for row in group
                if row["l0_correct"]
                and row["e1_correct"]
                and row["mask_group"] == "all_zero"
            ],
            per_class_stable_mask_group,
        )

    by_person = {str(row["person_key"]): row for row in candidates}
    queue = []
    for person_key, reasons in selected.items():
        row = dict(by_person[person_key])
        row["review_reasons"] = ";".join(sorted(reasons))
        queue.append(row)
    queue.sort(
        key=lambda row: (
            int(row["reference_id"]),
            -float(row["probability_l1_drift"]),
            str(row["person_key"]),
        )
    )
    for rank, row in enumerate(queue, start=1):
        row["review_rank"] = rank
    return queue
