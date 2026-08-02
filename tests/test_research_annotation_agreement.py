from copy import deepcopy

import pytest
import yaml

from src.research_annotation_agreement import (
    validate_and_summarize_completed_review,
    validate_review_template,
)
from src.research_annotation_review import PUBLIC_REVIEW_FIELDS


def _config():
    return yaml.safe_load(
        """
review_targets: [积液]
agreement:
  bootstrap_samples: 20
  bootstrap_seed: 7
  minimum_raw_presence_agreement: 0.70
  minimum_binary_positive_agreement: 0.60
  minimum_presence_kappa: 0.20
review_fields:
  presence_state: [present, absent_visible, not_in_view, uncertain]
  image_mode: [B, PD, CD, unknown]
  annotation_scope: [exhaustive_for_declared_targets, positive_only, uncertain]
  polygon_action: [keep, adjust, add, remove, not_applicable]
  subtype_by_target:
    积液: [joint_effusion, other_fluid, uncertain, not_applicable]
"""
    )


def _row(index, left="present", right="present"):
    row = {field: "" for field in PUBLIC_REVIEW_FIELDS}
    row.update(
        {
            "review_case_key": f"r{index}",
            "image_key": f"i{index}",
            "target_category": "积液",
            "required_independent_reviews": "2",
            "reviewer_1_presence_state": left,
            "reviewer_1_image_mode": "B",
            "reviewer_1_annotation_scope": "exhaustive_for_declared_targets",
            "reviewer_1_polygon_action": "keep",
            "reviewer_1_subtype": "joint_effusion",
            "reviewer_2_presence_state": right,
            "reviewer_2_image_mode": "B",
            "reviewer_2_annotation_scope": "exhaustive_for_declared_targets",
            "reviewer_2_polygon_action": "keep",
            "reviewer_2_subtype": "joint_effusion",
        }
    )
    if left != right:
        row.update(
            {
                "adjudicated_presence_state": "present",
                "adjudication_notes": "consensus review",
            }
        )
    return row


def test_completed_review_reports_agreement_and_binary_positive_agreement():
    rows = [
        _row(1),
        _row(2, "absent_visible", "absent_visible"),
        _row(3, "present", "absent_visible"),
        _row(4, "uncertain", "uncertain"),
    ]

    result = validate_and_summarize_completed_review(rows, _config())

    summary = result["targets"]["积液"]
    assert summary["raw_presence_agreement"] == 0.75
    assert summary["binary_comparable_rows"] == 3
    assert summary["binary_positive_agreement"] == pytest.approx(2 / 3)
    assert summary["binary_negative_agreement"] == pytest.approx(2 / 3)
    assert result["interpretation"]["only_absent_visible_is_negative"] is True
    assert result["preregistered_gates"]["all_targets_passed"] is True


def test_disagreement_requires_adjudication():
    row = _row(1, "present", "absent_visible")
    row["adjudicated_presence_state"] = ""
    row["adjudication_notes"] = ""

    with pytest.raises(ValueError, match="Missing adjudication"):
        validate_and_summarize_completed_review([row], _config())


def test_template_rejects_duplicate_images():
    first = _row(1)
    second = deepcopy(_row(2))
    second["image_key"] = first["image_key"]

    with pytest.raises(ValueError, match="images must be unique"):
        validate_review_template([first, second], {"积液"})


def test_template_rejects_raw_paths_in_free_text():
    row = _row(1)
    row["reviewer_1_notes"] = "workspace/data/raw/hidden/image.jpg"

    with pytest.raises(ValueError, match="identity/path"):
        validate_review_template([row], {"积液"})
