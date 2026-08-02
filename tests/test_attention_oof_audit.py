from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "30_audit_attention_oof.py"
SPEC = importlib.util.spec_from_file_location("audit_attention_oof", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_rows_pools_patients_and_folds():
    rows = [
        {"person_key": "p1", "image_key": "i1", "outer_fold": "0", "image_count": "2", "attention_weight": "0.96"},
        {"person_key": "p1", "image_key": "i2", "outer_fold": "0", "image_count": "2", "attention_weight": "0.04"},
        {"person_key": "p2", "image_key": "i3", "outer_fold": "1", "image_count": "1", "attention_weight": "1.0"},
        {"person_key": "p3", "image_key": "i4", "outer_fold": "1", "image_count": "2", "attention_weight": "0.6"},
        {"person_key": "p3", "image_key": "i5", "outer_fold": "1", "image_count": "2", "attention_weight": "0.4"},
    ]

    result = MODULE.audit_rows(rows, collapse_threshold=0.95, max_collapse_rate=0.5)

    assert result["contract"]["patients"] == 3
    assert result["contract"]["unique_images"] == 5
    assert result["pooled"]["multi_image_patients"] == 2
    assert result["pooled"]["multi_image_collapse_rate"] == 0.5
    assert result["pooled"]["collapse_gate_passed"] is True
    assert set(result["by_fold"]) == {"0", "1"}


def test_audit_rows_rejects_invalid_patient_weight_sum():
    rows = [
        {"person_key": "p1", "image_key": "i1", "outer_fold": "0", "image_count": "2", "attention_weight": "0.7"},
        {"person_key": "p1", "image_key": "i2", "outer_fold": "0", "image_count": "2", "attention_weight": "0.2"},
    ]

    with pytest.raises(ValueError, match="weight_sum=1"):
        MODULE.audit_rows(rows, collapse_threshold=0.95, max_collapse_rate=0.5)
