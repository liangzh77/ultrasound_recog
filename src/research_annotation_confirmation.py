"""Fail-closed contract for S1a clinical decisions before preregistration."""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
ADR_REFERENCE = re.compile(r"^docs/decisions/ADR-\d+-.+\.md$")

EXPECTED_MEDICAL_OPTIONS = {
    "Q1": ("滑膜", ("synovial_hypertrophy", "anatomy_and_abnormal_mixed", "other")),
    "Q2": (
        "滑膜血流",
        ("pd_synovial_only", "pd_cd_separate_synovial_only", "other"),
    ),
    "Q3": ("结晶", ("split_omeract_like", "single_crystal_class", "other")),
    "Q4": (
        "软骨",
        ("normal_structure", "cartilage_damage", "split_structure_damage", "other"),
    ),
    "Q5": (
        "骨皮质",
        ("anatomy_surface_only", "includes_cortical_abnormality", "other"),
    ),
    "Q6": (
        "囊肿与滑囊层级",
        ("hierarchy_with_subtypes", "all_independent", "single_merged_class", "other"),
    ),
    "Q7": (
        "旧标注穷尽性",
        ("all_28_exhaustive", "positive_only", "per_target_scope", "unknown"),
    ),
    "Q8": ("诊断盲法复核", ("fully_blinded", "partially_blinded", "not_blinded")),
}
RECOMMENDED_MEDICAL_OPTIONS = {
    "Q1": "synovial_hypertrophy",
    "Q2": "pd_cd_separate_synovial_only",
    "Q3": "split_omeract_like",
    "Q4": "split_structure_damage",
    "Q5": "anatomy_surface_only",
    "Q6": "hierarchy_with_subtypes",
    "Q7": "per_target_scope",
    "Q8": "fully_blinded",
}
EXPECTED_PARAMETERS = {
    "P1": ("total_review_images", 400),
    "P2": ("independent_reviewers_per_image", 2),
    "P3": ("double_review_fraction", 1.0),
    "P4": ("adjudicate_every_core_disagreement", True),
    "P5": ("minimum_raw_presence_agreement", 0.80),
    "P6": ("minimum_binary_positive_agreement", 0.85),
    "P7": ("minimum_presence_kappa", 0.60),
    "P8": (
        "bootstrap_and_gate_basis",
        {"bootstrap_samples": 2000, "ci_level": 0.95, "gate_basis": "point_estimate"},
    ),
}
EXPECTED_STUDY_ID = "knee_patient_multimodal_v1_20260724"
REVIEW_TARGET_COUNT = 8
OUTER_FOLDS = 5
SELECTION_BUCKETS = 2
TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "study",
    "provenance",
    "medical_decisions",
    "review_parameters",
    "signoffs",
    "deviation_decision_reference",
}
PROVENANCE_FIELDS = {
    "study_id",
    "dataset_fingerprint",
    "annotation_version",
    "source_git_commit",
    "source_workbook_sha256",
    "completed_workbook_sha256",
}
MEDICAL_FIELDS = {
    "topic",
    "status",
    "allowed_options",
    "selected_option",
    "operational_definition",
    "decision_reason",
}
PARAMETER_FIELDS = {
    "parameter",
    "status",
    "recommended_value",
    "final_value",
    "decision_reason",
}
SIGNOFF_FIELDS = {
    "clinical_role",
    "clinical_confirmation_present",
    "research_role",
    "research_confirmation_present",
    "confirmation_date",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "patient_name",
    "person_key",
    "image_key",
    "raw_path",
    "raw_image",
    "raw_annotation",
    "姓名",
)
FORBIDDEN_VALUE_FRAGMENTS = (
    "workspace/data/raw/",
    "/private/",
    "patient_name",
    "person_key",
    "image_key",
    "raw_image_path",
    "raw_annotation_path",
    "姓名",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _all_items(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _all_items(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_items(item)


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def _validate_privacy(payload: Mapping[str, Any]) -> None:
    for key, _ in _all_items(payload):
        normalized_key = key.casefold()
        if any(fragment in normalized_key for fragment in FORBIDDEN_KEY_FRAGMENTS):
            raise ValueError("Clinical confirmation contains a forbidden identity field")
    for value in _all_strings(payload):
        normalized = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
            raise ValueError("Clinical confirmation contains an absolute path")
        if any(fragment in normalized.casefold() for fragment in FORBIDDEN_VALUE_FRAGMENTS):
            raise ValueError("Clinical confirmation contains a forbidden identity/path value")


def _require_exact_fields(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{label} fields differ from the frozen contract")


def _equivalent_decision(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent_decision(left[key], right[key]) for key in left
        )
    return type(left) is type(right) and left == right


def _validate_structure(
    payload: Mapping[str, Any],
    expected_dataset_fingerprint: str,
    expected_annotation_version: str,
    expected_source_workbook_sha256: str,
) -> None:
    _require_exact_fields(payload, TOP_LEVEL_FIELDS, "Top-level confirmation")
    if payload["schema_version"] != 0:
        raise ValueError("Unsupported clinical confirmation schema version")
    if payload["study"] != "knee_annotation_review_confirmation_s1a":
        raise ValueError("Unexpected clinical confirmation study")
    provenance = payload["provenance"]
    _require_exact_fields(provenance, PROVENANCE_FIELDS, "Confirmation provenance")
    if provenance["study_id"] != EXPECTED_STUDY_ID:
        raise ValueError("Clinical confirmation study ID mismatch")
    if provenance["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("Clinical confirmation dataset fingerprint mismatch")
    if provenance["annotation_version"] != expected_annotation_version:
        raise ValueError("Clinical confirmation annotation version mismatch")
    if provenance["source_workbook_sha256"] != expected_source_workbook_sha256:
        raise ValueError("Clinical confirmation source workbook SHA-256 mismatch")
    if not SHA256.fullmatch(_clean(provenance["dataset_fingerprint"])):
        raise ValueError("Dataset fingerprint must be a lowercase SHA-256")
    if not SHA256.fullmatch(_clean(provenance["source_workbook_sha256"])):
        raise ValueError("Source workbook fingerprint must be a lowercase SHA-256")
    completed_workbook_sha256 = provenance["completed_workbook_sha256"]
    if completed_workbook_sha256 is not None and not SHA256.fullmatch(
        _clean(completed_workbook_sha256)
    ):
        raise ValueError("Completed workbook fingerprint must be a lowercase SHA-256")
    if not GIT_COMMIT.fullmatch(_clean(provenance["source_git_commit"])):
        raise ValueError("Source Git commit must be a 40-character lowercase hash")

    medical = payload["medical_decisions"]
    if set(medical) != set(EXPECTED_MEDICAL_OPTIONS):
        raise ValueError("Clinical confirmation must contain exactly Q1-Q8")
    for question_id, (topic, options) in EXPECTED_MEDICAL_OPTIONS.items():
        decision = medical[question_id]
        _require_exact_fields(decision, MEDICAL_FIELDS, question_id)
        if decision["topic"] != topic:
            raise ValueError(f"{question_id} topic differs from the frozen contract")
        if tuple(decision["allowed_options"]) != options:
            raise ValueError(f"{question_id} options differ from the frozen contract")

    parameters = payload["review_parameters"]
    if set(parameters) != set(EXPECTED_PARAMETERS):
        raise ValueError("Clinical confirmation must contain exactly P1-P8")
    for parameter_id, (name, recommended) in EXPECTED_PARAMETERS.items():
        parameter = parameters[parameter_id]
        _require_exact_fields(parameter, PARAMETER_FIELDS, parameter_id)
        if parameter["parameter"] != name:
            raise ValueError(f"{parameter_id} name differs from the frozen contract")
        if not _equivalent_decision(parameter["recommended_value"], recommended):
            raise ValueError(f"{parameter_id} recommendation differs from the frozen contract")

    _require_exact_fields(payload["signoffs"], SIGNOFF_FIELDS, "Signoffs")
    reference = payload["deviation_decision_reference"]
    if reference is not None and not ADR_REFERENCE.fullmatch(_clean(reference)):
        raise ValueError("Deviation decision reference must point to a project ADR")
    _validate_privacy(payload)


def validate_confirmation_template(
    payload: Mapping[str, Any],
    expected_dataset_fingerprint: str,
    expected_annotation_version: str,
    expected_source_workbook_sha256: str,
) -> dict[str, Any]:
    """Validate that an unfilled template is complete and fail-closed."""
    _validate_structure(
        payload,
        expected_dataset_fingerprint,
        expected_annotation_version,
        expected_source_workbook_sha256,
    )
    if payload["status"] != "draft_unconfirmed":
        raise ValueError("Template status must be draft_unconfirmed")
    if payload["provenance"]["completed_workbook_sha256"] is not None:
        raise ValueError("Draft template cannot claim a completed workbook")
    for decision in payload["medical_decisions"].values():
        if decision["status"] != "pending" or any(
            decision[field] is not None
            for field in ("selected_option", "operational_definition", "decision_reason")
        ):
            raise ValueError("Draft medical decisions must remain unfilled")
    for parameter in payload["review_parameters"].values():
        if parameter["status"] != "pending" or any(
            parameter[field] is not None for field in ("final_value", "decision_reason")
        ):
            raise ValueError("Draft review parameters must remain unfilled")
    if payload["signoffs"] != {
        "clinical_role": None,
        "clinical_confirmation_present": False,
        "research_role": None,
        "research_confirmation_present": False,
        "confirmation_date": None,
    }:
        raise ValueError("Draft signoffs must remain empty")
    if payload["deviation_decision_reference"] is not None:
        raise ValueError("Draft template cannot preregister an unseen deviation")
    return {
        "status": "draft_confirmation_template_contract_passed",
        "medical_questions": len(EXPECTED_MEDICAL_OPTIONS),
        "review_parameters": len(EXPECTED_PARAMETERS),
        "privacy_contract_passed": True,
        "ready_for_preregistration": False,
    }


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} is outside the allowed range")
    return numeric


def _validate_parameter_values(parameters: Mapping[str, Any]) -> dict[str, Any]:
    values = {key: item["final_value"] for key, item in parameters.items()}
    if isinstance(values["P1"], bool) or not isinstance(values["P1"], int):
        raise ValueError("P1 total review images must be an integer")
    if not 1 <= values["P1"] <= 10000:
        raise ValueError("P1 total review images is outside the allowed range")
    if isinstance(values["P2"], bool) or not isinstance(values["P2"], int):
        raise ValueError("P2 independent reviewers must be an integer")
    if not 2 <= values["P2"] <= 5:
        raise ValueError("P2 independent reviewers is outside the allowed range")
    _number(values["P3"], "P3 double review fraction", 0.01, 1.0)
    if not isinstance(values["P4"], bool):
        raise ValueError("P4 adjudication decision must be boolean")
    for parameter_id in ("P5", "P6", "P7"):
        _number(values[parameter_id], parameter_id, 0.0, 1.0)
    bootstrap = values["P8"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "bootstrap_samples",
        "ci_level",
        "gate_basis",
    }:
        raise ValueError("P8 bootstrap decision differs from the frozen contract")
    if isinstance(bootstrap["bootstrap_samples"], bool) or not isinstance(
        bootstrap["bootstrap_samples"], int
    ):
        raise ValueError("P8 bootstrap samples must be an integer")
    if not 500 <= bootstrap["bootstrap_samples"] <= 100000:
        raise ValueError("P8 bootstrap samples is outside the allowed range")
    _number(bootstrap["ci_level"], "P8 CI level", 0.80, 0.999)
    if bootstrap["gate_basis"] not in {"point_estimate", "ci_lower_bound"}:
        raise ValueError("P8 gate basis is unsupported")
    return values


def _validate_confirmation_date(value: Any) -> None:
    try:
        date.fromisoformat(_clean(value))
    except ValueError as error:
        raise ValueError("Confirmation date must be YYYY-MM-DD") from error


def validate_completed_confirmation(
    payload: Mapping[str, Any],
    expected_dataset_fingerprint: str,
    expected_annotation_version: str,
    expected_source_workbook_sha256: str,
    expected_completed_workbook_sha256: str,
) -> dict[str, Any]:
    """Validate completed decisions and report whether the current code can freeze them."""
    _validate_structure(
        payload,
        expected_dataset_fingerprint,
        expected_annotation_version,
        expected_source_workbook_sha256,
    )
    if payload["status"] != "completed_clinical_confirmation":
        raise ValueError("Completed confirmation status is required")
    if (
        payload["provenance"]["completed_workbook_sha256"]
        != expected_completed_workbook_sha256
    ):
        raise ValueError("Completed workbook SHA-256 does not match the returned file")
    if not SHA256.fullmatch(expected_completed_workbook_sha256):
        raise ValueError("Expected completed workbook SHA-256 is invalid")

    selections = {}
    for question_id, decision in payload["medical_decisions"].items():
        if decision["status"] != "confirmed":
            raise ValueError(f"{question_id} is not confirmed")
        selected = _clean(decision["selected_option"])
        if selected not in decision["allowed_options"]:
            raise ValueError(f"{question_id} selected option is invalid")
        if not _clean(decision["operational_definition"]):
            raise ValueError(f"{question_id} operational definition is required")
        if not _clean(decision["decision_reason"]):
            raise ValueError(f"{question_id} decision reason is required")
        selections[question_id] = selected

    for parameter_id, parameter in payload["review_parameters"].items():
        if parameter["status"] != "confirmed":
            raise ValueError(f"{parameter_id} is not confirmed")
        if parameter["final_value"] is None:
            raise ValueError(f"{parameter_id} final value is required")
        if not _clean(parameter["decision_reason"]):
            raise ValueError(f"{parameter_id} decision reason is required")
    parameter_values = _validate_parameter_values(payload["review_parameters"])

    signoffs = payload["signoffs"]
    if not _clean(signoffs["clinical_role"]):
        raise ValueError("Clinical signoff role is required")
    if signoffs["clinical_confirmation_present"] is not True:
        raise ValueError("Clinical confirmation signoff is required")
    if not _clean(signoffs["research_role"]):
        raise ValueError("Research signoff role is required")
    if signoffs["research_confirmation_present"] is not True:
        raise ValueError("Research confirmation signoff is required")
    _validate_confirmation_date(signoffs["confirmation_date"])

    deviations = [
        question_id
        for question_id, selected in selections.items()
        if selected != RECOMMENDED_MEDICAL_OPTIONS[question_id]
    ]
    for parameter_id, (_, recommended) in EXPECTED_PARAMETERS.items():
        if not _equivalent_decision(parameter_values[parameter_id], recommended):
            deviations.append(parameter_id)

    contract_changes = []
    total_images = int(parameter_values["P1"])
    queue_divisor = REVIEW_TARGET_COUNT * SELECTION_BUCKETS * OUTER_FOLDS
    if total_images % queue_divisor:
        contract_changes.append("P1_requires_balanced_queue_generator_change")
    if int(parameter_values["P2"]) != 2:
        contract_changes.append("P2_requires_additional_reviewer_columns")
    if float(parameter_values["P3"]) != 1.0:
        contract_changes.append("P3_requires_partial_double_review_contract")
    if parameter_values["P4"] is not True:
        contract_changes.append("P4_conflicts_with_mandatory_adjudication")
    bootstrap = parameter_values["P8"]
    if float(bootstrap["ci_level"]) != 0.95:
        contract_changes.append("P8_requires_configurable_CI_level")
    if bootstrap["gate_basis"] != "point_estimate":
        contract_changes.append("P8_requires_CI_lower_bound_gate")

    deviation_reference = payload["deviation_decision_reference"]
    missing_deviation_adr = bool(deviations and not deviation_reference)
    ready = not contract_changes and not missing_deviation_adr
    result = {
        "status": "completed_confirmation_validated",
        "medical_questions_confirmed": len(selections),
        "review_parameters_confirmed": len(parameter_values),
        "privacy_contract_passed": True,
        "selected_options": selections,
        "method_deviations_from_recommendation": deviations,
        "deviation_decision_reference": deviation_reference,
        "missing_deviation_adr": missing_deviation_adr,
        "contract_changes_required": contract_changes,
        "ready_for_preregistration": ready,
    }
    if ready:
        per_fold_per_bucket = total_images // queue_divisor
        result["proposed_review_config_patch"] = {
            "status": "frozen_preregistered",
            "required_independent_reviews": int(parameter_values["P2"]),
            "agreement": {
                "bootstrap_samples": int(bootstrap["bootstrap_samples"]),
                "minimum_raw_presence_agreement": float(parameter_values["P5"]),
                "minimum_binary_positive_agreement": float(parameter_values["P6"]),
                "minimum_presence_kappa": float(parameter_values["P7"]),
                "require_adjudication_for_any_core_disagreement": True,
            },
            "selection": {
                "per_target_existing_positive": per_fold_per_bucket * 5,
                "per_target_legacy_unlabeled_candidate": per_fold_per_bucket * 5,
                "per_fold_per_bucket": per_fold_per_bucket,
            },
        }
    return result
