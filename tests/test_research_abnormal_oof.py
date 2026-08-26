from __future__ import annotations

import numpy as np

from src.research_abnormal_oof import D0OOFData, evaluate_d0_oof
from src.research_metrics import compute_patient_metrics


CLASSES = ("类风湿性关节炎", "痛风性关节炎", "脊柱关节炎", "骨性关节炎", "损伤")


def _data(probabilities: np.ndarray) -> D0OOFData:
    rows = tuple(
        {
            "person_key": f"P{index}",
            "outer_fold": str(index),
            "reference_id": str(index),
            "reference_class": CLASSES[index],
        }
        for index in range(5)
    )
    return D0OOFData(
        rows=rows,
        person_keys=tuple(row["person_key"] for row in rows),
        outer_folds=np.arange(5),
        targets=np.arange(5),
        probabilities=probabilities,
    )


def _config() -> dict:
    baseline = _baseline_probabilities()
    metrics = compute_patient_metrics(np.arange(5), baseline, CLASSES)
    return {
        "evaluation": {
            "bootstrap_samples": 20,
            "bootstrap_seed": 7,
            "baseline": {
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "multiclass_brier": metrics["multiclass_brier"],
            },
        },
        "gates": {
            "minimum_macro_f1": 0.39,
            "minimum_macro_f1_delta_vs_e2_conditional": 0.03,
            "minimum_positive_folds": 4,
            "maximum_per_class_f1_drop": 0.05,
            "minimum_each_class_f1": 0.20,
            "maximum_ece": 0.30,
            "maximum_multiclass_brier": 0.90,
            "maximum_multi_image_attention_collapse_rate": 0.50,
        },
    }


def _baseline_probabilities() -> np.ndarray:
    return np.roll(np.eye(5), 1, axis=1) * 0.6 + 0.08


def test_d0_gate_advances_only_when_every_check_passes() -> None:
    candidate = _data(np.eye(5) * 0.9 + 0.02)
    baseline = _data(_baseline_probabilities())
    report = evaluate_d0_oof(
        _config(),
        candidate,
        baseline,
        {"pooled": {"multi_image_collapse_rate": 0.1}},
        {"all_passed": True},
    )
    assert report["gate"]["all_passed"] is True
    assert report["gate"]["decision"] == "advance_to_f0_f1"
    assert report["d0_minus_e2_conditional"]["positive_folds"] == 5


def test_d0_gate_stops_on_attention_failure() -> None:
    candidate = _data(np.eye(5) * 0.9 + 0.02)
    baseline = _data(_baseline_probabilities())
    report = evaluate_d0_oof(
        _config(),
        candidate,
        baseline,
        {"pooled": {"multi_image_collapse_rate": 0.6}},
        {"all_passed": True},
    )
    assert report["gate"]["checks"]["attention_collapse"] is False
    assert report["gate"]["decision"] == "stop_without_fusion_training"
