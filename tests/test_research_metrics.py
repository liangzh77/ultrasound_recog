import numpy as np
import pytest

from src.research_metrics import (
    bootstrap_macro_auc_ci,
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
    validate_oof_predictions,
)


def test_oof_validation_requires_patient_level_unique_complete_probabilities():
    keys = ["P1", "P2", "P3"]
    targets = np.array([0, 1, 2])
    probabilities = np.eye(3)

    validate_oof_predictions(
        keys,
        targets,
        probabilities,
        expected_person_keys=set(keys),
        prediction_level="patient",
    )

    with pytest.raises(ValueError, match="unique"):
        validate_oof_predictions(
            ["P1", "P1", "P3"],
            targets,
            probabilities,
            expected_person_keys=set(keys),
            prediction_level="patient",
        )
    with pytest.raises(ValueError, match="patient-level"):
        validate_oof_predictions(
            keys,
            targets,
            probabilities,
            expected_person_keys=set(keys),
            prediction_level="image",
        )
    with pytest.raises(ValueError, match="sum to 1"):
        validate_oof_predictions(
            keys,
            targets,
            probabilities * 0.5,
            expected_person_keys=set(keys),
            prediction_level="patient",
        )


def test_perfect_patient_predictions_have_perfect_metrics_and_calibration():
    targets = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[targets]

    result = compute_patient_metrics(targets, probabilities, ("A", "B", "C"))

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["top2_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["macro_auc"] == 1.0
    assert result["multiclass_brier"] == 0.0
    assert result["ece"] == 0.0
    assert all(row["sensitivity"] == 1.0 for row in result["per_class"])
    assert all(row["specificity"] == 1.0 for row in result["per_class"])
    assert all(row["precision"] == 1.0 for row in result["per_class"])
    assert all(row["npv"] == 1.0 for row in result["per_class"])


def test_bootstrap_confidence_interval_is_reproducible():
    targets = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )

    first = bootstrap_macro_f1_ci(targets, probabilities, 100, seed=17)
    second = bootstrap_macro_f1_ci(targets, probabilities, 100, seed=17)

    assert first == second


def test_macro_auc_bootstrap_confidence_interval_is_reproducible():
    targets = np.asarray([0, 0, 1, 1, 2, 2])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ]
    )

    first = bootstrap_macro_auc_ci(targets, probabilities, 20, seed=17)
    second = bootstrap_macro_auc_ci(targets, probabilities, 20, seed=17)

    assert first == second
    assert first[0] <= first[1] <= first[2]


def test_paired_bootstrap_compares_models_on_identical_patients():
    targets = np.array([0, 0, 1, 1, 2, 2])
    strong = np.eye(3)[targets] * 0.9 + 0.1 / 3
    weak = np.full((6, 3), 1 / 3)

    result = paired_bootstrap_macro_f1(
        targets,
        baseline_probabilities=weak,
        candidate_probabilities=strong,
        n_bootstrap=100,
        seed=23,
    )

    assert result["observed_delta"] > 0
    assert result["ci_low"] > 0
    assert result["probability_delta_gt_zero"] == 1.0
