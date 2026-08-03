from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.research_dataset import ResearchImageRecord
from src.research_gate import (
    ABNORMAL_DIAGNOSES,
    GATE_CLASSES,
    apply_temperature,
    bootstrap_roc_auc_ci,
    build_gate_prediction_rows,
    compute_gate_metrics,
    diagnosis_to_gate_id,
    fit_temperature,
    fit_gate_postprocessor,
    load_gate_config,
    remap_records_to_gate,
    select_operating_threshold,
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


def test_threshold_is_fitted_only_on_inner_validation_and_meets_sensitivity():
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    abnormal = np.asarray([0.10, 0.20, 0.60, 0.70, 0.80, 0.90])
    probabilities = np.column_stack((1 - abnormal, abnormal))

    selected = select_operating_threshold(
        targets,
        probabilities,
        minimum_abnormal_sensitivity=0.90,
        fit_split="inner_validation",
    )

    assert selected.threshold == 0.70
    assert selected.abnormal_sensitivity == 1.0
    assert selected.normal_specificity == 1.0
    assert selected.constraint_met is True
    with pytest.raises(ValueError, match="only be fitted on inner_validation"):
        select_operating_threshold(
            targets,
            probabilities,
            minimum_abnormal_sensitivity=0.90,
            fit_split="outer_test",
        )


def test_gate_metrics_are_patient_level_and_block_all_abnormal_shortcut():
    targets = np.asarray([0, 0, 1, 1])
    abnormal = np.asarray([0.1, 0.2, 0.8, 0.9])
    probabilities = np.column_stack((1 - abnormal, abnormal))

    perfect = compute_gate_metrics(targets, probabilities, threshold=0.5)
    all_abnormal = compute_gate_metrics(targets, probabilities, threshold=0.0)

    assert perfect["roc_auc"] == 1.0
    assert perfect["pr_auc"] == 1.0
    assert perfect["macro_f1"] == 1.0
    assert perfect["abnormal_sensitivity"] == 1.0
    assert perfect["normal_specificity"] == 1.0
    assert perfect["confusion_matrix"] == [[2, 0], [0, 2]]
    assert all_abnormal["normal_specificity"] == 0.0
    assert all_abnormal["macro_f1"] < 0.70


def test_gate_input_contract_rejects_one_class_and_bad_probability_sums():
    with pytest.raises(ValueError, match="both normal and abnormal"):
        compute_gate_metrics(
            np.asarray([1, 1]),
            np.asarray([[0.2, 0.8], [0.1, 0.9]]),
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="sum to 1"):
        compute_gate_metrics(
            np.asarray([0, 1]),
            np.asarray([[0.2, 0.7], [0.1, 0.8]]),
            threshold=0.5,
        )


def test_stratified_bootstrap_auc_is_reproducible_and_contains_perfect_auc():
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    abnormal = np.asarray([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    probabilities = np.column_stack((1 - abnormal, abnormal))

    first = bootstrap_roc_auc_ci(targets, probabilities, samples=100, seed=7)
    second = bootstrap_roc_auc_ci(targets, probabilities, samples=100, seed=7)

    assert first == second == (1.0, 1.0, 1.0)


def test_temperature_is_fitted_only_on_inner_validation_and_improves_nll():
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray(
        [
            [0.60, 0.40],
            [0.65, 0.35],
            [0.70, 0.30],
            [0.40, 0.60],
            [0.35, 0.65],
            [0.30, 0.70],
        ]
    )

    calibration = fit_temperature(
        targets, probabilities, fit_split="inner_validation"
    )
    calibrated = apply_temperature(probabilities, calibration.temperature)

    assert 0.05 <= calibration.temperature < 1.0
    assert calibration.validation_nll_after < calibration.validation_nll_before
    assert calibration.used_identity_fallback is False
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.all(calibrated[np.arange(len(targets)), targets] > 0.60)
    with pytest.raises(ValueError, match="only be fitted on inner_validation"):
        fit_temperature(targets, probabilities, fit_split="outer_test")


def test_temperature_application_rejects_invalid_temperature():
    probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]])

    with pytest.raises(ValueError, match="finite and positive"):
        apply_temperature(probabilities, 0.0)


def test_postprocessor_fits_calibration_before_threshold_and_builds_safe_rows():
    targets = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray(
        [[0.65, 0.35], [0.70, 0.30], [0.40, 0.60], [0.30, 0.70]]
    )
    postprocessor = fit_gate_postprocessor(
        targets,
        probabilities,
        minimum_abnormal_sensitivity=0.90,
    )

    rows = build_gate_prediction_rows(
        person_keys=["P2", "P1", "P4", "P3"],
        targets=targets,
        probabilities=probabilities,
        image_counts=[2, 1, 3, 2],
        outer_fold=2,
        model_id="G0-fold2-test",
        postprocessor=postprocessor,
    )

    assert [row["person_key"] for row in rows] == ["P1", "P2", "P3", "P4"]
    assert {row["prediction_level"] for row in rows} == {"patient_gate"}
    assert {row["reference_class"] for row in rows} == {"normal", "abnormal"}
    assert all(abs(row["prob_normal"] + row["prob_abnormal"] - 1) < 1e-12 for row in rows)
    assert all(row["temperature"] == postprocessor.calibration.temperature for row in rows)
    forbidden = {"diagnosis", "raw_path", "image_path", "filename"}
    assert not forbidden.intersection(rows[0])


def test_gate_prediction_rows_reject_duplicate_patients():
    targets = np.asarray([0, 1])
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    postprocessor = fit_gate_postprocessor(
        targets,
        probabilities,
        minimum_abnormal_sensitivity=0.90,
    )

    with pytest.raises(ValueError, match="person_key values must be unique"):
        build_gate_prediction_rows(
            person_keys=["P1", "P1"],
            targets=targets,
            probabilities=probabilities,
            image_counts=[1, 1],
            outer_fold=0,
            model_id="G0-fold0-test",
            postprocessor=postprocessor,
        )
