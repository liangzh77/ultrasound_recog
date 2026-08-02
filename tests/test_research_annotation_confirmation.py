from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.research_annotation_confirmation import (
    RECOMMENDED_MEDICAL_OPTIONS,
    validate_completed_confirmation,
    validate_confirmation_template,
)


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "configs/research/annotation_clinical_confirmation_v0.yaml"


def _template():
    return yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))


def _expected(payload):
    provenance = payload["provenance"]
    return {
        "expected_dataset_fingerprint": provenance["dataset_fingerprint"],
        "expected_annotation_version": provenance["annotation_version"],
        "expected_source_workbook_sha256": provenance[
            "source_workbook_sha256"
        ],
    }


def _completed():
    payload = _template()
    payload["status"] = "completed_clinical_confirmation"
    payload["provenance"]["completed_workbook_sha256"] = "1" * 64
    for question_id, decision in payload["medical_decisions"].items():
        decision["status"] = "confirmed"
        decision["selected_option"] = RECOMMENDED_MEDICAL_OPTIONS[question_id]
        decision["operational_definition"] = f"definition for {question_id}"
        decision["decision_reason"] = "accepted after clinical review"
    for parameter in payload["review_parameters"].values():
        parameter["status"] = "confirmed"
        parameter["final_value"] = deepcopy(parameter["recommended_value"])
        parameter["decision_reason"] = "accepted before review"
    payload["signoffs"] = {
        "clinical_role": "musculoskeletal_ultrasound_clinician",
        "clinical_confirmation_present": True,
        "research_role": "study_lead",
        "research_confirmation_present": True,
        "confirmation_date": "2026-08-03",
    }
    return payload


def test_draft_template_is_complete_private_and_not_ready_to_freeze():
    payload = _template()

    result = validate_confirmation_template(payload, **_expected(payload))

    assert result["medical_questions"] == 8
    assert result["review_parameters"] == 8
    assert result["privacy_contract_passed"] is True
    assert result["ready_for_preregistration"] is False


def test_completed_recommended_contract_builds_read_only_config_patch():
    payload = _completed()

    result = validate_completed_confirmation(
        payload,
        expected_completed_workbook_sha256="1" * 64,
        **_expected(payload),
    )

    assert result["ready_for_preregistration"] is True
    assert result["method_deviations_from_recommendation"] == []
    patch = result["proposed_review_config_patch"]
    assert patch["status"] == "frozen_preregistered"
    assert patch["selection"]["per_fold_per_bucket"] == 5
    assert patch["agreement"]["minimum_presence_kappa"] == 0.60


def test_completed_confirmation_rejects_missing_medical_definition():
    payload = _completed()
    payload["medical_decisions"]["Q3"]["operational_definition"] = ""

    with pytest.raises(ValueError, match="Q3 operational definition"):
        validate_completed_confirmation(
            payload,
            expected_completed_workbook_sha256="1" * 64,
            **_expected(payload),
        )


def test_confirmation_rejects_private_identity_or_path_fields():
    payload = _template()
    payload["patient_name"] = "hidden"

    with pytest.raises(ValueError, match="fields differ"):
        validate_confirmation_template(payload, **_expected(payload))

    payload = _template()
    payload["medical_decisions"]["Q1"]["decision_reason"] = (
        "C:/private/patient/image.jpg"
    )
    with pytest.raises(ValueError, match="absolute path"):
        validate_confirmation_template(payload, **_expected(payload))


def test_current_two_reviewer_contract_reports_incompatible_decisions():
    payload = _completed()
    payload["review_parameters"]["P2"]["final_value"] = 3
    payload["review_parameters"]["P3"]["final_value"] = 0.5

    result = validate_completed_confirmation(
        payload,
        expected_completed_workbook_sha256="1" * 64,
        **_expected(payload),
    )

    assert result["ready_for_preregistration"] is False
    assert result["contract_changes_required"] == [
        "P2_requires_additional_reviewer_columns",
        "P3_requires_partial_double_review_contract",
    ]
    assert "proposed_review_config_patch" not in result


def test_deviation_requires_adr_before_preregistration():
    payload = _completed()
    payload["medical_decisions"]["Q4"]["selected_option"] = "normal_structure"
    payload["review_parameters"]["P7"]["final_value"] = 0.50

    result = validate_completed_confirmation(
        payload,
        expected_completed_workbook_sha256="1" * 64,
        **_expected(payload),
    )

    assert result["method_deviations_from_recommendation"] == ["Q4", "P7"]
    assert result["missing_deviation_adr"] is True
    assert result["ready_for_preregistration"] is False

    payload["deviation_decision_reference"] = (
        "docs/decisions/ADR-011-example-clinical-deviation.md"
    )
    result = validate_completed_confirmation(
        payload,
        expected_completed_workbook_sha256="1" * 64,
        deviation_reference_verified=True,
        **_expected(payload),
    )
    assert result["missing_deviation_adr"] is False
    assert result["ready_for_preregistration"] is True


def test_process_parameter_ranges_are_validated():
    payload = _completed()
    payload["review_parameters"]["P6"]["final_value"] = 1.1

    with pytest.raises(ValueError, match="P6 is outside"):
        validate_completed_confirmation(
            payload,
            expected_completed_workbook_sha256="1" * 64,
            **_expected(payload),
        )


def test_signoffs_accept_role_codes_not_names():
    payload = _completed()
    payload["signoffs"]["clinical_role"] = "Dr Zhang"

    with pytest.raises(ValueError, match="Clinical signoff role is invalid"):
        validate_completed_confirmation(
            payload,
            expected_completed_workbook_sha256="1" * 64,
            **_expected(payload),
        )


def test_completed_confirmation_binds_the_returned_workbook_hash():
    payload = _completed()

    with pytest.raises(ValueError, match="does not match the returned file"):
        validate_completed_confirmation(
            payload,
            expected_completed_workbook_sha256="2" * 64,
            **_expected(payload),
        )


def test_completed_confirmation_rejects_the_unchanged_blank_workbook():
    payload = _completed()
    source_sha = payload["provenance"]["source_workbook_sha256"]
    payload["provenance"]["completed_workbook_sha256"] = source_sha

    with pytest.raises(ValueError, match="must differ from the blank source"):
        validate_completed_confirmation(
            payload,
            expected_completed_workbook_sha256=source_sha,
            **_expected(payload),
        )


def test_template_rejects_version_mismatch():
    payload = _template()

    with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
        validate_confirmation_template(
            payload,
            expected_dataset_fingerprint="0" * 64,
            expected_annotation_version=payload["provenance"]["annotation_version"],
            expected_source_workbook_sha256=payload["provenance"][
                "source_workbook_sha256"
            ],
        )
