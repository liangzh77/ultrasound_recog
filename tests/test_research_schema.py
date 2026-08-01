import pytest

from src.research_schema import (
    DIAGNOSIS_CLASSES,
    EXCLUDED_DIAGNOSES,
    normalize_join_key,
    validate_model_feature_columns,
)


def test_primary_diagnosis_order_is_fixed_and_excludes_synovial_cyst():
    assert DIAGNOSIS_CLASSES == (
        "正常",
        "类风湿性关节炎",
        "痛风性关节炎",
        "脊柱关节炎",
        "骨性关节炎",
        "损伤",
    )
    assert EXCLUDED_DIAGNOSES == frozenset({"滑膜囊肿"})


def test_excel_join_keys_are_case_and_whitespace_insensitive():
    assert normalize_join_key(" spa35\t") == "SPA35"
    assert normalize_join_key("损伤 1") == "损伤1"


def test_model_feature_allowlist_rejects_leakage_columns():
    validate_model_feature_columns(["age_years", "sex", "crp_mg_l"])

    with pytest.raises(ValueError, match="诊断"):
        validate_model_feature_columns(["age_years", "诊断"])

    with pytest.raises(ValueError, match="patient_folder"):
        validate_model_feature_columns(["patient_folder"])
