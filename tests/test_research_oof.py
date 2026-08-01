import csv
import json
from pathlib import Path

import pytest

from src.research_oof import evaluate_oof_file


CLASSES = (
    "正常",
    "类风湿性关节炎",
    "痛风性关节炎",
    "脊柱关节炎",
    "骨性关节炎",
    "损伤",
)
PROBABILITY_COLUMNS = (
    "prob_normal",
    "prob_ra",
    "prob_ga",
    "prob_spa",
    "prob_oa",
    "prob_injury",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _registry(tmp_path: Path) -> Path:
    registry = tmp_path / "registry"
    registry.mkdir()
    patients = []
    folds = []
    for class_id, diagnosis in enumerate(CLASSES):
        for duplicate in range(2):
            person_key = f"P{class_id}{duplicate}"
            patients.append(
                {
                    "person_key": person_key,
                    "diagnosis": diagnosis,
                    "diagnosis_id": class_id,
                    "include": 1,
                }
            )
            folds.append(
                {
                    "person_key": person_key,
                    "diagnosis": diagnosis,
                    "diagnosis_id": class_id,
                    "outer_fold": duplicate,
                }
            )
    _write_csv(registry / "patients.csv", patients)
    _write_csv(registry / "folds_outer.csv", folds)
    (registry / "reference_standard.json").write_text(
        json.dumps({"classes": list(CLASSES)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return registry


def _predictions(tmp_path: Path, wrong_fold: bool = False) -> Path:
    rows = []
    for class_id, diagnosis in enumerate(CLASSES):
        for duplicate in range(2):
            probabilities = [0.02] * len(CLASSES)
            probabilities[class_id] = 0.90
            row = {
                "prediction_level": "patient",
                "person_key": f"P{class_id}{duplicate}",
                "outer_fold": 4 if wrong_fold and class_id == 0 and duplicate == 0 else duplicate,
                "reference_class": diagnosis,
                "reference_id": class_id,
                "image_count": 3,
                "model_id": "TEST",
            }
            row.update(dict(zip(PROBABILITY_COLUMNS, probabilities, strict=True)))
            rows.append(row)
    path = tmp_path / "oof.csv"
    _write_csv(path, rows)
    return path


def test_evaluate_oof_file_validates_registry_and_returns_patient_metrics(tmp_path):
    result = evaluate_oof_file(
        prediction_path=_predictions(tmp_path),
        registry_dir=_registry(tmp_path),
        n_bootstrap=20,
        seed=17,
    )

    assert result["contract"]["prediction_level"] == "patient"
    assert result["contract"]["patients"] == 12
    assert result["metrics"]["macro_f1"] == 1.0
    assert result["macro_f1_95_ci"] == [1.0, 1.0, 1.0]
    assert set(result["fold_metrics"]) == {"0", "1"}
    assert result["fold_summary"]["macro_f1"] == {
        "mean": 1.0,
        "standard_deviation": 0.0,
    }
    assert "prediction_path" not in json.dumps(result)


def test_evaluate_oof_file_rejects_outer_fold_mismatch(tmp_path):
    with pytest.raises(ValueError, match="outer_fold mismatch"):
        evaluate_oof_file(
            prediction_path=_predictions(tmp_path, wrong_fold=True),
            registry_dir=_registry(tmp_path),
            n_bootstrap=5,
            seed=17,
        )
