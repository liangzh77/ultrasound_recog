from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.research_ledger import (
    validate_experiment_record,
    validate_research_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "research" / "experiment_ledger.yaml"


def test_current_research_ledger_has_required_provenance_and_valid_hashes():
    result = validate_research_ledger(LEDGER, ROOT, verify_artifacts=True)

    assert result["study_id"] == "knee_patient_multimodal_v1_20260724"
    assert result["experiments"] == 9
    assert result["formal_models"] == 4


def test_formal_model_cannot_omit_oof_or_configuration():
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    record = deepcopy(ledger["experiments"][0])
    record.pop("oof")

    with pytest.raises(ValueError, match="oof"):
        validate_experiment_record(record, ROOT, verify_artifacts=False)


def test_ledger_rejects_absolute_or_raw_patient_paths():
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    record = deepcopy(ledger["experiments"][0])
    record["markdown_report"] = "C:/private/patient.jpg"

    with pytest.raises(ValueError, match="absolute|raw/private"):
        validate_experiment_record(record, ROOT, verify_artifacts=False)


def test_in_progress_formal_model_verifies_every_completed_fold_oof():
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    record = deepcopy(
        next(
            item
            for item in ledger["experiments"]
            if item["id"] == "E1S-fivefold-formal"
        )
    )
    record["additional_fold_oof"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="Additional OOF"):
        validate_experiment_record(record, ROOT, verify_artifacts=True)


def test_nonformal_attempt_result_artifacts_are_hash_verified():
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    record = deepcopy(
        next(
            item
            for item in ledger["experiments"]
            if item["id"] == "E2-fold0-formal-attempt1-resource-stop"
        )
    )
    record["results"]["stopped_summary_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="Result artifact mismatch"):
        validate_experiment_record(record, ROOT, verify_artifacts=True)
