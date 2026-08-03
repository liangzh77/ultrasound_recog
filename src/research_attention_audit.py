"""Reusable patient-level attention OOF loading and concentration audit."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.research_ledger import sha256_file
from src.research_mil import summarize_attention


REQUIRED_COLUMNS = {
    "person_key",
    "image_key",
    "outer_fold",
    "image_count",
    "attention_weight",
}


def read_attention_files(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    inputs: list[dict] = []
    seen_images: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
        for row in rows:
            image_key = row["image_key"]
            if image_key in seen_images:
                raise ValueError(f"duplicate image_key across attention files: {image_key}")
            seen_images.add(image_key)
            all_rows.append(row)
        inputs.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )
    return all_rows, inputs


def audit_attention_rows(
    rows: list[dict],
    *,
    collapse_threshold: float,
    max_collapse_rate: float,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["person_key"]].append(row)

    summaries: list[dict] = []
    folds: dict[int, list[dict]] = defaultdict(list)
    weight_sum_errors: list[str] = []
    image_count_errors: list[str] = []
    person_fold_errors: list[str] = []
    for person_key, person_rows in grouped.items():
        weights = [float(row["attention_weight"]) for row in person_rows]
        declared_counts = {int(row["image_count"]) for row in person_rows}
        outer_folds = {int(row["outer_fold"]) for row in person_rows}
        if abs(sum(weights) - 1.0) > 1e-5:
            weight_sum_errors.append(person_key)
        if declared_counts != {len(person_rows)}:
            image_count_errors.append(person_key)
        if len(outer_folds) != 1:
            person_fold_errors.append(person_key)
            continue
        summary = {"attention_weights": weights}
        summaries.append(summary)
        folds[next(iter(outer_folds))].append(summary)

    if weight_sum_errors or image_count_errors or person_fold_errors:
        raise ValueError(
            "attention contract failed: "
            f"weight_sum={len(weight_sum_errors)}, "
            f"image_count={len(image_count_errors)}, "
            f"person_fold={len(person_fold_errors)}"
        )

    pooled = summarize_attention(summaries, collapse_threshold=collapse_threshold)
    pooled["max_allowed_multi_image_collapse_rate"] = max_collapse_rate
    pooled["collapse_gate_passed"] = bool(
        pooled["multi_image_collapse_rate"] <= max_collapse_rate
    )
    by_fold = {}
    for fold, fold_summaries in sorted(folds.items()):
        fold_audit = summarize_attention(
            fold_summaries,
            collapse_threshold=collapse_threshold,
        )
        fold_audit["collapse_gate_passed"] = bool(
            fold_audit["multi_image_collapse_rate"] <= max_collapse_rate
        )
        by_fold[str(fold)] = fold_audit
    return {
        "contract": {
            "prediction_level": "image_attention",
            "unique_images": len(rows),
            "patients": len(grouped),
            "weight_sum_tolerance": 1e-5,
            "weight_sum_errors": 0,
            "image_count_errors": 0,
            "person_fold_errors": 0,
        },
        "pooled": pooled,
        "by_fold": by_fold,
    }
