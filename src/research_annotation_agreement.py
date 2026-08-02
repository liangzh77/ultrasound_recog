"""Validation and agreement statistics for blinded clinical annotation review."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.research_annotation_review import (
    FORBIDDEN_PUBLIC_FIELDS,
    PUBLIC_REVIEW_FIELDS,
)


CORE_REVIEW_FIELDS = (
    "presence_state",
    "image_mode",
    "annotation_scope",
    "polygon_action",
    "subtype",
)
FORBIDDEN_REVIEW_VALUE_FRAGMENTS = (
    "workspace/data/raw/",
    "/private/",
    "raw_image_path",
    "raw_annotation_path",
    "patient_name",
    "姓名",
)


def validate_review_template(
    rows: Iterable[Mapping[str, Any]],
    expected_targets: set[str],
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("Review queue is empty")
    for record in records:
        if set(record) != set(PUBLIC_REVIEW_FIELDS):
            raise ValueError("Review queue fields differ from the blinded contract")
        if set(record) & FORBIDDEN_PUBLIC_FIELDS:
            raise ValueError("Review queue contains a forbidden public field")
        for value in record.values():
            normalized = _clean(value).replace("\\", "/")
            if re.match(r"^[A-Za-z]:/", normalized) or any(
                fragment in normalized
                for fragment in FORBIDDEN_REVIEW_VALUE_FRAGMENTS
            ):
                raise ValueError("Review queue contains a forbidden identity/path value")
    case_keys = [str(row["review_case_key"]) for row in records]
    image_keys = [str(row["image_key"]) for row in records]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Review case keys must be unique")
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("Review images must be unique")
    observed_targets = {str(row["target_category"]) for row in records}
    if not observed_targets <= expected_targets:
        raise ValueError("Review queue contains an unexpected target")
    if observed_targets != expected_targets:
        raise ValueError("Review queue does not cover every configured target")
    review_counts = {
        int(_clean(row["required_independent_reviews"])) for row in records
    }
    if len(review_counts) != 1 or next(iter(review_counts)) < 2:
        raise ValueError("Review queue must require at least two independent reviews")
    return {
        "rows": len(records),
        "unique_cases": len(set(case_keys)),
        "unique_images": len(set(image_keys)),
        "targets": sorted(observed_targets),
        "privacy_contract_passed": True,
    }


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _kappa(left: list[str], right: list[str], labels: list[str]) -> float | None:
    value = float(cohen_kappa_score(left, right, labels=labels))
    return value if np.isfinite(value) else None


def _presence_statistics(
    left: list[str],
    right: list[str],
    presence_states: list[str],
) -> dict[str, Any]:
    pairs = Counter(zip(left, right))
    total = len(left)
    raw_agreement = sum(a == b for a, b in zip(left, right)) / total
    binary_pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a in {"present", "absent_visible"}
        and b in {"present", "absent_visible"}
    ]
    both_positive = sum(a == b == "present" for a, b in binary_pairs)
    both_negative = sum(a == b == "absent_visible" for a, b in binary_pairs)
    discordant = sum(a != b for a, b in binary_pairs)
    positive_denominator = 2 * both_positive + discordant
    negative_denominator = 2 * both_negative + discordant
    return {
        "rows": total,
        "raw_presence_agreement": raw_agreement,
        "presence_kappa": _kappa(left, right, presence_states),
        "presence_confusion": [
            {"reviewer_1": a, "reviewer_2": b, "count": count}
            for (a, b), count in sorted(pairs.items())
        ],
        "binary_comparable_rows": len(binary_pairs),
        "binary_positive_agreement": (
            2 * both_positive / positive_denominator
            if positive_denominator
            else None
        ),
        "binary_negative_agreement": (
            2 * both_negative / negative_denominator
            if negative_denominator
            else None
        ),
    }


def _bootstrap_intervals(
    left: list[str],
    right: list[str],
    presence_states: list[str],
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    generator = np.random.default_rng(seed)
    observed: dict[str, list[float]] = {
        "raw_presence_agreement_ci95": [],
        "presence_kappa_ci95": [],
        "binary_positive_agreement_ci95": [],
        "binary_negative_agreement_ci95": [],
    }
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    for _ in range(samples):
        indices = generator.integers(0, len(left), size=len(left))
        values = _presence_statistics(
            left_array[indices].tolist(),
            right_array[indices].tolist(),
            presence_states,
        )
        mapping = {
            "raw_presence_agreement_ci95": values["raw_presence_agreement"],
            "presence_kappa_ci95": values["presence_kappa"],
            "binary_positive_agreement_ci95": values[
                "binary_positive_agreement"
            ],
            "binary_negative_agreement_ci95": values[
                "binary_negative_agreement"
            ],
        }
        for key, value in mapping.items():
            if value is not None and np.isfinite(value):
                observed[key].append(float(value))
    result = {}
    for key, values in observed.items():
        if not values:
            result[key] = None
        else:
            low, high = np.percentile(values, (2.5, 97.5))
            result[key] = [float(low), float(high)]
    return result


def validate_and_summarize_completed_review(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    targets = list(config["review_targets"])
    validate_review_template(records, set(targets))
    fields = config["review_fields"]
    allowed = {
        "presence_state": set(fields["presence_state"]),
        "image_mode": set(fields["image_mode"]),
        "annotation_scope": set(fields["annotation_scope"]),
        "polygon_action": set(fields["polygon_action"]),
    }
    subtype_by_target = {
        target: set(values)
        for target, values in fields["subtype_by_target"].items()
    }
    incomplete = []
    adjudication_missing = []
    final_presence_counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        target = _clean(row["target_category"])
        for reviewer in ("reviewer_1", "reviewer_2"):
            for field in CORE_REVIEW_FIELDS:
                value = _clean(row[f"{reviewer}_{field}"])
                permitted = (
                    subtype_by_target[target]
                    if field == "subtype"
                    else allowed[field]
                )
                if value not in permitted:
                    incomplete.append(str(row["review_case_key"]))
                    break
        disagreements = [
            field
            for field in CORE_REVIEW_FIELDS
            if _clean(row[f"reviewer_1_{field}"])
            != _clean(row[f"reviewer_2_{field}"])
        ]
        if disagreements:
            for field in disagreements:
                value = _clean(row[f"adjudicated_{field}"])
                permitted = (
                    subtype_by_target[target]
                    if field == "subtype"
                    else allowed[field]
                )
                if value not in permitted:
                    adjudication_missing.append(str(row["review_case_key"]))
                    break
            if not _clean(row["adjudication_notes"]):
                adjudication_missing.append(str(row["review_case_key"]))
        presence_disagrees = "presence_state" in disagreements
        final_presence = (
            _clean(row["adjudicated_presence_state"])
            if presence_disagrees
            else _clean(row["reviewer_1_presence_state"])
        )
        if final_presence:
            final_presence_counts[(target, final_presence)] += 1
    if incomplete:
        raise ValueError(
            f"Incomplete or invalid double review in {len(set(incomplete))} cases"
        )
    if adjudication_missing:
        raise ValueError(
            f"Missing adjudication in {len(set(adjudication_missing))} cases"
        )

    agreement_config = config["agreement"]
    summaries = {}
    for target in targets:
        target_rows = [
            row for row in records if _clean(row["target_category"]) == target
        ]
        left = [_clean(row["reviewer_1_presence_state"]) for row in target_rows]
        right = [_clean(row["reviewer_2_presence_state"]) for row in target_rows]
        summary = _presence_statistics(left, right, list(fields["presence_state"]))
        summary["confidence_intervals"] = {
            "method": "row_bootstrap_percentile",
            "samples": int(agreement_config["bootstrap_samples"]),
            "seed": int(agreement_config["bootstrap_seed"]),
            **_bootstrap_intervals(
                left,
                right,
                list(fields["presence_state"]),
                int(agreement_config["bootstrap_samples"]),
                int(agreement_config["bootstrap_seed"]),
            ),
        }
        summaries[target] = summary

    thresholds = {
        "raw_presence_agreement": agreement_config[
            "minimum_raw_presence_agreement"
        ],
        "binary_positive_agreement": agreement_config[
            "minimum_binary_positive_agreement"
        ],
        "presence_kappa": agreement_config["minimum_presence_kappa"],
    }
    if any(not isinstance(value, (int, float)) for value in thresholds.values()):
        raise ValueError("Agreement gates must be numeric in the frozen config")
    gates = {}
    for target, summary in summaries.items():
        checks = {
            metric: summary[metric] is not None
            and float(summary[metric]) >= float(threshold)
            for metric, threshold in thresholds.items()
        }
        gates[target] = {"checks": checks, "passed": all(checks.values())}

    return {
        "status": "completed_double_review_and_adjudication_validated",
        "rows": len(records),
        "targets": summaries,
        "preregistered_gates": {
            "thresholds": thresholds,
            "by_target": gates,
            "all_targets_passed": all(item["passed"] for item in gates.values()),
        },
        "final_presence_counts": [
            {"target": target, "presence_state": state, "count": count}
            for (target, state), count in sorted(final_presence_counts.items())
        ],
        "interpretation": {
            "legacy_unlabeled_is_negative": False,
            "only_absent_visible_is_negative": True,
            "kappa_must_be_interpreted_with_prevalence_sensitive_agreements": True,
        },
    }
