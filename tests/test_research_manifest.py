from src.research_manifest import (
    extract_identity_token,
    pseudonymous_key,
    resolve_identity_tokens,
    validate_roi,
)


def test_identity_uses_original_source_folder_basename():
    annotation = {
        "info": {
            "folder": r"C:\source\膝关节2024\外伤\901196782_某患者",
        }
    }

    assert extract_identity_token(annotation) == "901196782_某患者"


def test_pseudonymous_key_is_deterministic_and_hides_identity():
    first = pseudonymous_key("901196782_某患者", b"fixed-test-key", "KNEE_DEV")
    second = pseudonymous_key("901196782_某患者", b"fixed-test-key", "KNEE_DEV")

    assert first == second
    assert first.startswith("KNEE_DEV_")
    assert "某患者" not in first


def test_roi_requires_positive_in_bounds_rectangle():
    assert validate_roi(
        {"x1": 10, "y1": 5, "x2": 90, "y2": 70},
        width=100,
        height=80,
    )
    assert not validate_roi(
        {"x1": 90, "y1": 5, "x2": 10, "y2": 70},
        width=100,
        height=80,
    )
    assert not validate_roi(
        {"x1": 10, "y1": 5, "x2": 101, "y2": 70},
        width=100,
        height=80,
    )


def test_folder_pseudonym_is_treated_as_alias_when_original_identity_exists():
    result = resolve_identity_tokens(
        {"901195048_某患者", "RA186"},
        patient_folder="RA186",
    )

    assert result == "901195048_某患者"
