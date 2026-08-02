"""Fail-closed, independent reviewer response contract for S1a."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from src.research_annotation_agreement import validate_review_template
from src.research_annotation_review import PUBLIC_REVIEW_FIELDS


FORMAL_ENTRY_PROTOCOL = "semantic_presence_only"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REVIEWER_VALUE_FIELDS = (
    "presence_state",
    "image_mode",
    "annotation_scope",
    "polygon_action",
    "subtype",
    "notes",
)
REQUIRED_SEMANTIC_FIELDS = (
    "presence_state",
    "image_mode",
    "annotation_scope",
    "subtype",
)
BASE_FIELDS = (
    "review_case_key",
    "image_key",
    "target_category",
    "required_independent_reviews",
)
ADJUDICATION_FIELDS = tuple(
    field for field in PUBLIC_REVIEW_FIELDS if field.startswith("adjudicat")
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def reviewer_prefix(slot: int) -> str:
    if slot not in {1, 2}:
        raise ValueError("Reviewer slot must be 1 or 2")
    return f"reviewer_{slot}"


def validate_formal_entry_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse formal entry unless the clinical preregistration is frozen."""
    if config.get("status") != "frozen_preregistered":
        raise ValueError("Formal review entry requires a frozen_preregistered config")
    if config.get("study") != "knee_annotation_review_queue_s1a":
        raise ValueError("Formal review config has an unexpected study identifier")
    if not SHA256.fullmatch(_clean(config.get("dataset_fingerprint"))):
        raise ValueError("Formal review data fingerprint is invalid")
    if not re.fullmatch(
        r"annotations_[0-9a-f]{12}", _clean(config.get("annotation_version"))
    ):
        raise ValueError("Formal review annotation version is invalid")
    provenance = config.get("frozen_provenance", {})
    if not SHA256.fullmatch(
        _clean(provenance.get("clinical_confirmation_report_sha256"))
    ):
        raise ValueError("Formal review confirmation report hash is not frozen")
    if not SHA256.fullmatch(_clean(provenance.get("queue_sha256"))):
        raise ValueError("Formal review queue hash is not frozen")
    queue_rows = provenance.get("queue_rows")
    if isinstance(queue_rows, bool) or not isinstance(queue_rows, int) or queue_rows < 1:
        raise ValueError("Formal review queue row count is not frozen")
    if not GIT_COMMIT.fullmatch(_clean(provenance.get("review_workflow_git_commit"))):
        raise ValueError("Formal review workflow Git commit is not frozen")
    if int(config.get("required_independent_reviews", 0)) != 2:
        raise ValueError("Formal S1a entry currently requires exactly two reviewers")
    selection = config.get("selection", {})
    if selection.get("diagnosis_visible_to_reviewer") is not False:
        raise ValueError("Formal review must remain diagnosis blinded")
    if selection.get("legacy_annotation_visible_to_reviewer") is not False:
        raise ValueError("Formal S1a review cannot reveal legacy annotations")
    protocol = config.get("review_protocol", {})
    expected_protocol = {
        "stage": FORMAL_ENTRY_PROTOCOL,
        "legacy_polygon_visible_to_reviewer": False,
        "reviewer_polygon_capture": False,
        "polygon_action_policy": "fixed_not_applicable",
        "geometry_reliability_stage": "S1b",
    }
    if protocol != expected_protocol:
        raise ValueError("Formal S1a semantic review protocol is not frozen exactly")
    polygon_actions = set(
        config.get("review_fields", {}).get("polygon_action", [])
    )
    if "not_applicable" not in polygon_actions:
        raise ValueError("S1a semantic review requires not_applicable polygon action")
    return {
        "status": "formal_semantic_entry_contract_passed",
        "required_independent_reviews": 2,
        "diagnosis_blinded": True,
        "legacy_annotation_blinded": True,
        "geometry_capture_enabled": False,
        "geometry_reliability_stage": "S1b",
        "queue_sha256": provenance["queue_sha256"],
        "queue_rows": queue_rows,
        "review_workflow_git_commit": provenance["review_workflow_git_commit"],
    }


def _allowed_values(config: Mapping[str, Any], target: str) -> dict[str, set[str]]:
    fields = config["review_fields"]
    subtype_by_target = fields["subtype_by_target"]
    if target not in subtype_by_target:
        raise ValueError("Review target has no subtype contract")
    return {
        "presence_state": set(fields["presence_state"]),
        "image_mode": set(fields["image_mode"]),
        "annotation_scope": set(fields["annotation_scope"]),
        "polygon_action": {"not_applicable"},
        "subtype": set(subtype_by_target[target]),
    }


def new_reviewer_response_rows(
    queue_rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    slot: int,
) -> list[dict[str, str]]:
    """Create one reviewer-only response without exposing another response."""
    validate_formal_entry_config(config)
    prefix = reviewer_prefix(slot)
    rows = [dict(row) for row in queue_rows]
    validate_review_template(rows, set(config["review_targets"]))
    if len(rows) != int(config["frozen_provenance"]["queue_rows"]):
        raise ValueError("Review queue row count differs from frozen provenance")
    response = []
    for source in rows:
        if any(
            _clean(source[field])
            for field in PUBLIC_REVIEW_FIELDS
            if field not in BASE_FIELDS
        ):
            raise ValueError("New independent response requires a blank review queue")
        row = {field: _clean(source.get(field)) for field in PUBLIC_REVIEW_FIELDS}
        for reviewer_slot in (1, 2):
            reviewer = f"reviewer_{reviewer_slot}"
            for field in REVIEWER_VALUE_FIELDS:
                row[f"{reviewer}_{field}"] = ""
        for field in ADJUDICATION_FIELDS:
            row[field] = ""
        row[f"{prefix}_polygon_action"] = "not_applicable"
        response.append(row)
    return response


def validate_reviewer_response_rows(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    slot: int,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate one response file and prove reviewer independence."""
    validate_formal_entry_config(config)
    records = [dict(row) for row in rows]
    validate_review_template(records, set(config["review_targets"]))
    if len(records) != int(config["frozen_provenance"]["queue_rows"]):
        raise ValueError("Review response row count differs from frozen provenance")
    expected_reviews = str(config["required_independent_reviews"])
    if any(
        _clean(row["required_independent_reviews"]) != expected_reviews
        for row in records
    ):
        raise ValueError("Review response reviewer count differs from frozen config")
    own_prefix = reviewer_prefix(slot)
    other_prefix = reviewer_prefix(2 if slot == 1 else 1)
    complete = 0
    partial = 0
    for row in records:
        if any(_clean(row[field]) for field in ADJUDICATION_FIELDS):
            raise ValueError("Independent response cannot contain adjudication values")
        if any(
            _clean(row[f"{other_prefix}_{field}"])
            for field in REVIEWER_VALUE_FIELDS
        ):
            raise ValueError("Independent response exposes another reviewer slot")
        target = _clean(row["target_category"])
        allowed = _allowed_values(config, target)
        values = {
            field: _clean(row[f"{own_prefix}_{field}"])
            for field in REVIEWER_VALUE_FIELDS
        }
        started = any(values[field] for field in REQUIRED_SEMANTIC_FIELDS)
        semantic_complete = all(values[field] for field in REQUIRED_SEMANTIC_FIELDS)
        if values["polygon_action"] != "not_applicable":
            raise ValueError("S1a semantic response cannot submit polygon actions")
        for field in REQUIRED_SEMANTIC_FIELDS:
            value = values[field]
            if value and value not in allowed[field]:
                raise ValueError("Independent response contains an invalid coded value")
        if started and not semantic_complete:
            partial += 1
        elif semantic_complete:
            complete += 1
        if require_complete and not semantic_complete:
            raise ValueError("Independent response is incomplete")
    return {
        "status": (
            "complete_independent_response_validated"
            if require_complete
            else "independent_response_progress_validated"
        ),
        "reviewer_slot": slot,
        "rows": len(records),
        "completed_rows": complete,
        "partial_rows": partial,
        "remaining_rows": len(records) - complete,
        "other_reviewer_values_present": False,
        "adjudication_values_present": False,
        "diagnosis_or_legacy_annotation_present": False,
        "geometry_values_present": False,
    }


def merge_independent_reviewer_rows(
    reviewer_1_rows: Iterable[Mapping[str, Any]],
    reviewer_2_rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Merge two complete isolated responses for later blinded adjudication."""
    left = [dict(row) for row in reviewer_1_rows]
    right = [dict(row) for row in reviewer_2_rows]
    validate_reviewer_response_rows(left, config, 1, require_complete=True)
    validate_reviewer_response_rows(right, config, 2, require_complete=True)
    if len(left) != len(right):
        raise ValueError("Independent response row counts differ")
    merged = []
    for left_row, right_row in zip(left, right):
        if any(_clean(left_row[field]) != _clean(right_row[field]) for field in BASE_FIELDS):
            raise ValueError("Independent response queues differ")
        row = {field: _clean(left_row[field]) for field in PUBLIC_REVIEW_FIELDS}
        for field in REVIEWER_VALUE_FIELDS:
            row[f"reviewer_2_{field}"] = _clean(right_row[f"reviewer_2_{field}"])
        merged.append(row)
    return merged


def write_response_csv_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Atomically persist a validated reviewer response using the stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PUBLIC_REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
