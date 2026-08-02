from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_mil import summarize_attention  # noqa: E402


REQUIRED_COLUMNS = {
    "person_key",
    "image_key",
    "outer_fold",
    "image_count",
    "attention_weight",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def audit_rows(
    rows: list[dict], *, collapse_threshold: float, max_collapse_rate: float
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
            fold_summaries, collapse_threshold=collapse_threshold
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collapse-threshold", type=float, default=0.95)
    parser.add_argument("--max-collapse-rate", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, inputs = read_attention_files(args.fold_files)
    report = {
        "inputs": inputs,
        **audit_rows(
            rows,
            collapse_threshold=args.collapse_threshold,
            max_collapse_rate=args.max_collapse_rate,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
