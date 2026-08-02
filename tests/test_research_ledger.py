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
    assert result["experiments"] == 4
    assert result["formal_models"] == 2


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
