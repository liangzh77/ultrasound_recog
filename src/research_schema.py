"""Frozen labels and feature contracts for the 2026 patient-level study."""

from __future__ import annotations

import re
from collections.abc import Iterable


DIAGNOSIS_CLASSES = (
    "正常",
    "类风湿性关节炎",
    "痛风性关节炎",
    "脊柱关节炎",
    "骨性关节炎",
    "损伤",
)
DIAGNOSIS_TO_ID = {name: index for index, name in enumerate(DIAGNOSIS_CLASSES)}
EXCLUDED_DIAGNOSES = frozenset({"滑膜囊肿"})

CLINICAL_FEATURE_ALLOWLIST = frozenset(
    {
        "age_years",
        "sex",
        "esr_mm_h",
        "crp_mg_l",
        "anti_ccp_u_ml",
        "rf_iu_ml",
        "hla_b27",
        "uric_acid",
    }
)


def normalize_join_key(value: object) -> str:
    """Normalize folder/Excel identifiers without changing their meaning."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def validate_model_feature_columns(columns: Iterable[str]) -> None:
    invalid = sorted(set(columns) - CLINICAL_FEATURE_ALLOWLIST)
    if invalid:
        raise ValueError(
            "Columns outside the clinical feature allowlist: "
            + ", ".join(invalid)
        )
