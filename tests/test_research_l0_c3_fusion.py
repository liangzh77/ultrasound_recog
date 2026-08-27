from pathlib import Path

import numpy as np
import pytest

from src.research_l0_c3_fusion import L1Inputs, evaluate, load_config


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/research/l1_l0_c3_fixed_abnormal_fusion.yaml"


def test_l1_config_freezes_weights_and_forbids_2026():
    config = load_config(CONFIG)

    assert config["fixed_fusion"]["clinical_weight"] == 0.75
    assert config["fixed_fusion"]["image_weight"] == 0.25
    assert config["fixed_fusion"]["search_weights"] is False
    assert config["interpretation_contract"]["allow_2026"] is False


def test_l1_config_rejects_weight_search(tmp_path):
    content = CONFIG.read_text(encoding="utf-8").replace("search_weights: false", "search_weights: true")
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="weight search"):
        load_config(path)


def test_l1_evaluation_reports_paired_gate_without_training():
    targets = np.asarray([index % 5 for index in range(50)], dtype=np.int64)
    folds = np.asarray([index % 5 for index in range(50)], dtype=np.int64)
    c3 = np.full((50, 5), 0.05)
    c3[np.arange(50), targets] = 0.8
    l0 = c3.copy()
    data = L1Inputs(
        person_keys=tuple(f"P{index}" for index in range(50)),
        outer_folds=folds,
        targets=targets,
        reference_classes=tuple(str(value) for value in targets),
        l0_probabilities=l0,
        c3_probabilities=c3,
        fused_probabilities=0.75 * c3 + 0.25 * l0,
    )
    config = load_config(CONFIG)
    config["evaluation"]["bootstrap_samples"] = 20

    result = evaluate(config, data)

    assert result["patients"] == 50
    assert set(result["models"]) == {"L0_conditional", "C3", "L1_fixed_fusion"}
    assert result["fixed_fusion_minus_c3"]["observed_delta"] == 0.0
    assert result["decision_gate"]["passed"] is False
