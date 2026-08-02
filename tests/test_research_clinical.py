from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.research_clinical import (
    ALLOWED_MODEL_FEATURES,
    CLINICAL_CLASSES,
    build_logistic_pipeline,
    compare_clinical_oof,
    evaluate_clinical_oof,
    load_clinical_config,
    parse_hla_b27,
    parse_numeric_value,
    parse_sex,
    validate_source_headers,
    validate_clinical_oof,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/research/c0_c4_clinical_logreg.yaml"


def test_clinical_config_freezes_whitelist_and_experiment_order():
    config = load_clinical_config(CONFIG)
    assert tuple(config["feature_contract"]["allowed_model_features"]) == ALLOWED_MODEL_FEATURES
    assert tuple(config["experiments"]) == ("C0", "C1", "C2", "C3", "C4")


def test_clinical_config_rejects_diagnosis_as_model_feature(tmp_path):
    import yaml

    config = deepcopy(load_clinical_config(CONFIG))
    config["experiments"]["C3"]["features"].append("诊断")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="non-whitelisted|forbidden"):
        load_clinical_config(path)


@pytest.mark.parametrize(
    ("value", "expected", "censor"),
    [(10, 10.0, 0), ("<5", 5.0, -1), ("> 20.5", 20.5, 1), ("31岁", 31.0, 0), ("/", None, 0)],
)
def test_numeric_parser_preserves_boundary_without_raw_text(value, expected, censor):
    parsed, observed_censor, invalid = parse_numeric_value(value, ("", "/", "未查"))
    assert parsed == expected
    assert observed_censor == censor
    assert invalid is None


def test_nonnumeric_lab_value_is_audited_not_silently_encoded():
    parsed, censor, invalid = parse_numeric_value("阳性", ("", "/"))
    assert parsed is None and censor == 0 and invalid == "阳性"


def test_hla_and_sex_have_explicit_binary_contracts():
    assert parse_hla_b27("阳性", ("", "/")) == (1.0, None)
    assert parse_hla_b27("阴性", ("", "/")) == (0.0, None)
    assert parse_hla_b27("/", ("", "/")) == (None, None)
    assert parse_sex("男") == 1.0
    assert parse_sex("女") == 0.0


def test_source_header_contract_rejects_shifted_laboratory_columns():
    valid = [
        "编号",
        "性别",
        "年龄",
        "诊断",
        "病程",
        "超声检查日期",
        "血沉(mm/h)",
        "CRP(mg/L)",
        "ACCP（U/mL）",
        "RF(IU/mL）",
        "HLAB27",
        "尿酸",
    ]
    validate_source_headers(valid)
    invalid = list(valid)
    invalid[6], invalid[7] = invalid[7], invalid[6]
    with pytest.raises(ValueError, match="column 6"):
        validate_source_headers(invalid)


def test_preprocessing_fits_training_data_and_returns_all_classes():
    config = load_clinical_config(CONFIG)
    matrix = np.asarray([[0, 10], [1, 20], [0, np.nan], [1, 40], [0, 50]] * 5, dtype=float)
    targets = np.tile(np.arange(5), 5)
    pipeline = build_logistic_pipeline(config, 20260724)
    pipeline.fit(matrix, targets)
    probabilities = pipeline.predict_proba(matrix[:3])
    assert probabilities.shape == (3, len(CLINICAL_CLASSES))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_clinical_oof_rejects_duplicate_or_incomplete_patient_coverage():
    row = {
        "prediction_level": "patient_clinical",
        "person_key": "KNEE_DEV_TEST",
        "outer_fold": "0",
        "reference_class": CLINICAL_CLASSES[0],
        "reference_id": "0",
        "prob_ra": "1",
        "prob_ga": "0",
        "prob_spa": "0",
        "prob_oa": "0",
        "prob_injury": "0",
        "model_id": "test",
    }
    with pytest.raises(ValueError, match="exactly once"):
        validate_clinical_oof([row, dict(row)], ["KNEE_DEV_TEST"])


def test_paired_clinical_oof_rejects_changed_reference(tmp_path):
    import csv

    fields = [
        "prediction_level",
        "person_key",
        "outer_fold",
        "reference_class",
        "reference_id",
        "prob_ra",
        "prob_ga",
        "prob_spa",
        "prob_oa",
        "prob_injury",
        "model_id",
    ]
    baseline = {
        "prediction_level": "patient_clinical",
        "person_key": "KNEE_DEV_TEST",
        "outer_fold": 0,
        "reference_class": CLINICAL_CLASSES[0],
        "reference_id": 0,
        "prob_ra": 1,
        "prob_ga": 0,
        "prob_spa": 0,
        "prob_oa": 0,
        "prob_injury": 0,
        "model_id": "baseline",
    }
    candidate = dict(baseline, reference_id=1, reference_class=CLINICAL_CLASSES[1], model_id="candidate")
    paths = []
    for name, row in (("baseline.csv", baseline), ("candidate.csv", candidate)):
        path = tmp_path / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        paths.append(path)
    with pytest.raises(ValueError, match="reference targets differ"):
        compare_clinical_oof(paths[0], paths[1], ["KNEE_DEV_TEST"], 10, 1)


def test_clinical_oof_evaluation_returns_contract_metrics_and_ci(tmp_path):
    import csv

    fields = [
        "prediction_level",
        "person_key",
        "outer_fold",
        "reference_class",
        "reference_id",
        "prob_ra",
        "prob_ga",
        "prob_spa",
        "prob_oa",
        "prob_injury",
        "model_id",
    ]
    rows = []
    for class_id, class_name in enumerate(CLINICAL_CLASSES):
        probability = [0.025] * len(CLINICAL_CLASSES)
        probability[class_id] = 0.9
        rows.append(
            {
                "prediction_level": "patient_clinical",
                "person_key": f"KNEE_DEV_TEST_{class_id}",
                "outer_fold": class_id,
                "reference_class": class_name,
                "reference_id": class_id,
                **{
                    column: value
                    for column, value in zip(
                        ("prob_ra", "prob_ga", "prob_spa", "prob_oa", "prob_injury"),
                        probability,
                    )
                },
                "model_id": "test",
            }
        )
    path = tmp_path / "oof.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = evaluate_clinical_oof(
        path,
        [row["person_key"] for row in rows],
        bootstrap_samples=10,
        bootstrap_seed=1,
    )
    assert report["contract"]["patients"] == len(CLINICAL_CLASSES)
    assert report["metrics"]["macro_f1"] == 1.0
    assert report["macro_f1_95_ci"] == [1.0, 1.0, 1.0]
