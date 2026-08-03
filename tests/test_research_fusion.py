from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.research_fusion import load_x0_inputs, sha256_file


CLASSES = ("类风湿性关节炎", "痛风性关节炎", "脊柱关节炎", "骨性关节炎", "损伤")
PROB_COLUMNS = ("prob_ra", "prob_ga", "prob_spa", "prob_oa", "prob_injury")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> Path:
    image_rows = []
    clinical_rows = []
    for target, class_name in enumerate(CLASSES):
        values = np.full(5, 0.05)
        values[target] = 0.8
        image = {
            "prediction_level": "patient",
            "person_key": f"SAFE_{target}",
            "outer_fold": target,
            "reference_class": class_name,
            "reference_id": target + 1,
            "prob_normal": 0.1,
            "model_id": f"E2-fold{target}",
        }
        image.update(dict(zip(PROB_COLUMNS, values * 0.9)))
        clinical = {
            "prediction_level": "patient_clinical",
            "person_key": f"SAFE_{target}",
            "outer_fold": target,
            "reference_class": class_name,
            "reference_id": target,
            "model_id": f"C3-fold{target}",
        }
        clinical.update(dict(zip(PROB_COLUMNS, values)))
        image_rows.append(image)
        clinical_rows.append(clinical)
    image_path = tmp_path / "image.csv"
    clinical_path = tmp_path / "clinical.csv"
    _write_csv(image_path, image_rows)
    _write_csv(clinical_path, clinical_rows)
    config = {
        "study_code": "X0",
        "classes": ["ra", "ga", "spa", "oa", "injury"],
        "inputs": {
            "image_oof": {
                "path": image_path.name,
                "sha256": sha256_file(image_path),
                "probability_columns": list(PROB_COLUMNS),
            },
            "clinical_oof": {
                "path": clinical_path.name,
                "sha256": sha256_file(clinical_path),
                "probability_columns": list(PROB_COLUMNS),
            },
        },
        "cohort_contract": {
            "expected_patients": 5,
            "expected_outer_folds": [0, 1, 2, 3, 4],
        },
        "probability_contract": {"tolerance": 1e-6, "minimum_denominator": 1e-12},
        "primary_fixed_fusion": {
            "clinical_weight": 0.75,
            "image_weight": 0.25,
            "search_weights": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def test_load_x0_inputs_conditions_and_fuses_probabilities(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    _, data = load_x0_inputs(config_path, tmp_path)
    assert data.person_keys == tuple(f"SAFE_{index}" for index in range(5))
    assert np.allclose(data.image_probabilities.sum(axis=1), 1.0)
    assert np.allclose(data.fused_probabilities.sum(axis=1), 1.0)
    assert np.array_equal(data.targets, np.arange(5))


def test_load_x0_inputs_rejects_fold_mismatch(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    clinical_path = tmp_path / "clinical.csv"
    rows = list(csv.DictReader(clinical_path.open(encoding="utf-8-sig")))
    rows[0]["outer_fold"] = "4"
    _write_csv(clinical_path, rows)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["inputs"]["clinical_oof"]["sha256"] = sha256_file(clinical_path)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="fold or reference"):
        load_x0_inputs(config_path, tmp_path)


def test_load_x0_inputs_rejects_hash_mismatch(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["inputs"]["image_oof"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_x0_inputs(config_path, tmp_path)
