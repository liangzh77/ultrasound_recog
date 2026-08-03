from pathlib import Path

import pytest

from src.research_abnormal import (
    D0_CONFIG_SHA256,
    load_d0_config,
    remap_records_to_abnormal,
    validate_d0_record_sets,
)
from src.research_dataset import ResearchImageRecord


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/research/d0_roi_abnormal_fiveclass_mil_b2.yaml"


def _record(key: str, person: str, diagnosis: str, diagnosis_id: int) -> ResearchImageRecord:
    return ResearchImageRecord(
        image_key=key,
        person_key=person,
        diagnosis=diagnosis,
        diagnosis_id=diagnosis_id,
        image_path=Path("protected"),
        roi={"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
    )


def test_frozen_d0_config_loads_with_expected_hash() -> None:
    config = load_d0_config(CONFIG)
    assert config["model"]["num_classes"] == 5
    assert config["training"]["attention_kl_weight"] == 0.05
    import hashlib

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == D0_CONFIG_SHA256


def test_remap_records_excludes_normal_and_shifts_ids() -> None:
    records = [
        _record("n", "p0", "正常", 0),
        _record("r", "p1", "类风湿性关节炎", 1),
        _record("i", "p2", "损伤", 5),
    ]
    remapped = remap_records_to_abnormal(records)
    assert [(row.diagnosis, row.diagnosis_id) for row in remapped] == [
        ("类风湿性关节炎", 0),
        ("损伤", 4),
    ]


def test_remap_records_rejects_mismatched_source_id() -> None:
    with pytest.raises(ValueError, match="diagnosis and ID"):
        remap_records_to_abnormal([_record("r", "p1", "类风湿性关节炎", 2)])


def test_validate_d0_record_sets_checks_frozen_counts() -> None:
    records = remap_records_to_abnormal(
        [
            _record("r", "p1", "类风湿性关节炎", 1),
            _record("i", "p2", "损伤", 5),
        ]
    )
    config = {
        "data": {
            "expected_patients": 2,
            "expected_images": 2,
            "expected_patient_counts": {"类风湿性关节炎": 1, "损伤": 1},
            "expected_image_counts": {"类风湿性关节炎": 1, "损伤": 1},
        }
    }
    summary = validate_d0_record_sets(
        {"train": records[:1], "validation": [], "test": records[1:]}, config
    )
    assert summary["patients"] == 2
    assert summary["images"] == 2
