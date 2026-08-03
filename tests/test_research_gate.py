from copy import deepcopy
from pathlib import Path

import pytest

from src.research_dataset import ResearchImageRecord
from src.research_gate import (
    ABNORMAL_DIAGNOSES,
    GATE_CLASSES,
    diagnosis_to_gate_id,
    load_gate_config,
    remap_records_to_gate,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/research/g0_roi_normal_abnormal_gate_b2.yaml"


def test_frozen_g0_config_loads_with_expected_task_and_gates():
    config = load_gate_config(CONFIG)

    assert tuple(config["task"]["classes"]) == GATE_CLASSES
    assert config["data"]["expected_patients"] == 967
    assert config["gates"]["minimum_abnormal_sensitivity"] == 0.90
    assert config["privacy"]["allow_2026_labels"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "draft"),
        (("model", "num_classes"), 6),
        (("gates", "minimum_macro_f1"), 0.60),
        (("privacy", "allow_2026_labels"), True),
        (("runtime", "max_gpu_memory_gb"), 10.0),
    ],
)
def test_frozen_g0_config_rejects_contract_changes(tmp_path, path, value):
    import yaml

    config = deepcopy(load_gate_config(CONFIG))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen G0 config mismatch"):
        load_gate_config(candidate)


def test_all_six_diagnoses_map_to_the_frozen_binary_task():
    assert diagnosis_to_gate_id("正常") == 0
    assert all(diagnosis_to_gate_id(value) == 1 for value in ABNORMAL_DIAGNOSES)

    with pytest.raises(ValueError, match="outside the frozen G0 task"):
        diagnosis_to_gate_id("未知")


def test_record_remap_preserves_identity_roi_and_private_source(tmp_path):
    source = tmp_path / "private.png"
    records = [
        ResearchImageRecord(
            image_key="I001",
            person_key="P001",
            diagnosis="类风湿性关节炎",
            diagnosis_id=1,
            image_path=source,
            roi={"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        )
    ]

    remapped = remap_records_to_gate(records)

    assert remapped[0].diagnosis == "abnormal"
    assert remapped[0].diagnosis_id == 1
    assert remapped[0].image_key == records[0].image_key
    assert remapped[0].person_key == records[0].person_key
    assert remapped[0].image_path == source
    assert remapped[0].roi == records[0].roi
    assert records[0].diagnosis == "类风湿性关节炎"


def test_record_remap_rejects_inconsistent_source_diagnosis_id(tmp_path):
    records = [
        ResearchImageRecord(
            image_key="I001",
            person_key="P001",
            diagnosis="类风湿性关节炎",
            diagnosis_id=0,
            image_path=tmp_path / "private.png",
            roi={"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        )
    ]

    with pytest.raises(ValueError, match="diagnosis_id do not match"):
        remap_records_to_gate(records)
