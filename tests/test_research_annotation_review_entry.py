from copy import deepcopy
import csv

import pytest

from src.research_annotation_review import PUBLIC_REVIEW_FIELDS
from src.research_annotation_review_entry import (
    merge_independent_reviewer_rows,
    new_adjudication_rows,
    new_reviewer_response_rows,
    validate_adjudication_rows,
    validate_formal_entry_config,
    validate_reviewer_response_rows,
    write_response_csv_atomic,
)


def _config():
    return {
        "status": "frozen_preregistered",
        "study": "knee_annotation_review_queue_s1a",
        "dataset_fingerprint": "a" * 64,
        "annotation_version": "annotations_aaaaaaaaaaaa",
        "frozen_provenance": {
            "clinical_confirmation_report_sha256": "b" * 64,
            "queue_sha256": "c" * 64,
            "queue_rows": 1,
            "review_workflow_git_commit": "d" * 40,
        },
        "required_independent_reviews": 2,
        "selection": {
            "diagnosis_visible_to_reviewer": False,
            "legacy_annotation_visible_to_reviewer": False,
        },
        "review_protocol": {
            "stage": "semantic_presence_only",
            "legacy_polygon_visible_to_reviewer": False,
            "reviewer_polygon_capture": False,
            "polygon_action_policy": "fixed_not_applicable",
            "geometry_reliability_stage": "S1b",
        },
        "review_targets": ["积液"],
        "review_fields": {
            "presence_state": [
                "present",
                "absent_visible",
                "not_in_view",
                "uncertain",
            ],
            "image_mode": ["B", "PD", "CD", "unknown"],
            "annotation_scope": [
                "exhaustive_for_declared_targets",
                "positive_only",
                "uncertain",
            ],
            "polygon_action": [
                "keep",
                "adjust",
                "add",
                "remove",
                "not_applicable",
            ],
            "subtype_by_target": {
                "积液": [
                    "joint_effusion",
                    "other_fluid",
                    "uncertain",
                    "not_applicable",
                ]
            },
        },
    }


def _blank_queue():
    row = {field: "" for field in PUBLIC_REVIEW_FIELDS}
    row.update(
        {
            "review_case_key": "KNEE_REVIEW_TEST",
            "image_key": "KNEE_IMG_TEST",
            "target_category": "积液",
            "required_independent_reviews": "2",
        }
    )
    return [row]


def _complete(rows, slot):
    result = deepcopy(rows)
    prefix = f"reviewer_{slot}"
    result[0].update(
        {
            f"{prefix}_presence_state": "present",
            f"{prefix}_image_mode": "B",
            f"{prefix}_annotation_scope": "exhaustive_for_declared_targets",
            f"{prefix}_subtype": "joint_effusion",
        }
    )
    return result


def test_formal_entry_gate_rejects_unfrozen_or_geometry_enabled_config():
    config = _config()
    config["status"] = "draft_not_preregistered"
    with pytest.raises(ValueError, match="frozen_preregistered"):
        validate_formal_entry_config(config)

    config = _config()
    config["review_protocol"]["reviewer_polygon_capture"] = True
    with pytest.raises(ValueError, match="not frozen exactly"):
        validate_formal_entry_config(config)


def test_new_response_is_blank_and_isolates_one_reviewer():
    rows = new_reviewer_response_rows(_blank_queue(), _config(), 1)

    assert rows[0]["reviewer_1_polygon_action"] == "not_applicable"
    assert rows[0]["reviewer_2_polygon_action"] == ""
    result = validate_reviewer_response_rows(
        rows, _config(), 1, require_complete=False
    )
    assert result["remaining_rows"] == 1
    assert result["other_reviewer_values_present"] is False


def test_new_response_rejects_a_queue_containing_prior_answers():
    rows = _blank_queue()
    rows[0]["reviewer_2_presence_state"] = "present"

    with pytest.raises(ValueError, match="blank review queue"):
        new_reviewer_response_rows(rows, _config(), 1)


def test_response_validation_rejects_other_reviewer_and_polygon_values():
    rows = _complete(new_reviewer_response_rows(_blank_queue(), _config(), 1), 1)
    rows[0]["reviewer_2_presence_state"] = "absent_visible"
    with pytest.raises(ValueError, match="another reviewer"):
        validate_reviewer_response_rows(rows, _config(), 1, require_complete=True)

    rows[0]["reviewer_2_presence_state"] = ""
    rows[0]["reviewer_1_polygon_action"] = "keep"
    with pytest.raises(ValueError, match="cannot submit polygon"):
        validate_reviewer_response_rows(rows, _config(), 1, require_complete=True)


def test_two_complete_isolated_responses_merge_without_adjudication():
    reviewer_1 = _complete(
        new_reviewer_response_rows(_blank_queue(), _config(), 1), 1
    )
    reviewer_2 = _complete(
        new_reviewer_response_rows(_blank_queue(), _config(), 2), 2
    )
    reviewer_2[0]["reviewer_2_presence_state"] = "absent_visible"
    reviewer_2[0]["reviewer_2_subtype"] = "not_applicable"

    merged = merge_independent_reviewer_rows(reviewer_1, reviewer_2, _config())

    assert merged[0]["reviewer_1_presence_state"] == "present"
    assert merged[0]["reviewer_2_presence_state"] == "absent_visible"
    assert merged[0]["adjudicated_presence_state"] == ""


def test_atomic_csv_writer_preserves_stable_public_schema(tmp_path):
    rows = new_reviewer_response_rows(_blank_queue(), _config(), 1)
    output = tmp_path / "response.csv"

    write_response_csv_atomic(output, rows)

    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == PUBLIC_REVIEW_FIELDS
        assert list(reader)[0]["review_case_key"] == "KNEE_REVIEW_TEST"
    assert not (tmp_path / ".response.csv.tmp").exists()


def _merged_disagreement():
    reviewer_1 = _complete(
        new_reviewer_response_rows(_blank_queue(), _config(), 1), 1
    )
    reviewer_2 = _complete(
        new_reviewer_response_rows(_blank_queue(), _config(), 2), 2
    )
    reviewer_2[0]["reviewer_2_presence_state"] = "absent_visible"
    reviewer_2[0]["reviewer_2_subtype"] = "not_applicable"
    return merge_independent_reviewer_rows(reviewer_1, reviewer_2, _config())


def test_adjudication_requires_only_disagreed_fields_and_notes():
    rows = new_adjudication_rows(_merged_disagreement(), _config())
    progress = validate_adjudication_rows(rows, _config(), require_complete=False)
    assert progress["disagreement_rows"] == 1
    assert progress["remaining_rows"] == 1

    rows[0]["adjudicated_presence_state"] = "present"
    rows[0]["adjudicated_subtype"] = "joint_effusion"
    rows[0]["adjudication_notes"] = "consensus discussion"
    result = validate_adjudication_rows(rows, _config(), require_complete=True)
    assert result["completed_rows"] == 1


def test_adjudication_cannot_override_agreement_or_submit_geometry():
    rows = new_adjudication_rows(_merged_disagreement(), _config())
    rows[0]["adjudicated_image_mode"] = "PD"
    with pytest.raises(ValueError, match="cannot override"):
        validate_adjudication_rows(rows, _config(), require_complete=False)

    rows[0]["adjudicated_image_mode"] = ""
    rows[0]["adjudicated_polygon_action"] = "adjust"
    with pytest.raises(ValueError, match="cannot submit polygon"):
        validate_adjudication_rows(rows, _config(), require_complete=False)


def test_new_adjudication_rejects_prior_adjudication_values():
    rows = _merged_disagreement()
    rows[0]["adjudicated_presence_state"] = "present"

    with pytest.raises(ValueError, match="unadjudicated"):
        new_adjudication_rows(rows, _config())


def test_response_rejects_presence_subtype_contradiction():
    rows = _complete(new_reviewer_response_rows(_blank_queue(), _config(), 1), 1)
    rows[0]["reviewer_1_presence_state"] = "absent_visible"

    with pytest.raises(ValueError, match="not_applicable subtype"):
        validate_reviewer_response_rows(rows, _config(), 1, require_complete=True)


def test_adjudication_rejects_final_presence_subtype_contradiction():
    rows = new_adjudication_rows(_merged_disagreement(), _config())
    rows[0]["adjudicated_presence_state"] = "absent_visible"
    rows[0]["adjudicated_subtype"] = "joint_effusion"
    rows[0]["adjudication_notes"] = "consensus discussion"

    with pytest.raises(ValueError, match="not_applicable subtype"):
        validate_adjudication_rows(rows, _config(), require_complete=True)
