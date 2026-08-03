import csv
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

from src.research_gate import GATE_OOF_COLUMNS, load_gate_config
from src.research_gate_oof import (
    evaluate_gate_oof,
    load_and_validate_gate_oof,
    merge_gate_oof_fold_files,
    validate_gate_attention_alignment,
    validate_gate_fold_summaries,
)
from src.research_attention_audit import audit_attention_rows
from src.research_ledger import sha256_file
from src.research_schema import DIAGNOSIS_CLASSES


ROOT = Path(__file__).resolve().parent.parent


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_contract(tmp_path: Path):
    registry = tmp_path / "registry"
    people = []
    folds = []
    fold_paths = []
    for fold in range(5):
        fold_rows = []
        for target, diagnosis in ((0, "正常"), (1, "类风湿性关节炎")):
            person_key = f"P{fold}{target}"
            people.append(
                {"person_key": person_key, "diagnosis": diagnosis, "include": 1}
            )
            folds.append({"person_key": person_key, "outer_fold": fold})
            abnormal_probability = 0.05 if target == 0 else 0.95
            fold_rows.append(
                {
                    "prediction_level": "patient_gate",
                    "person_key": person_key,
                    "outer_fold": fold,
                    "reference_class": "normal" if target == 0 else "abnormal",
                    "reference_id": target,
                    "raw_prob_normal": 1 - abnormal_probability,
                    "raw_prob_abnormal": abnormal_probability,
                    "prob_normal": 1 - abnormal_probability,
                    "prob_abnormal": abnormal_probability,
                    "operating_threshold": 0.5,
                    "predicted_class": "normal" if target == 0 else "abnormal",
                    "predicted_id": target,
                    "temperature": 1.0,
                    "image_count": 1,
                    "model_id": f"G0-fold{fold}-seed{20260724 + fold}-formal",
                }
            )
        fold_path = tmp_path / f"fold{fold}.csv"
        _write_csv(fold_path, GATE_OOF_COLUMNS, fold_rows)
        fold_paths.append(fold_path)
    registry.mkdir()
    (registry / "reference_standard.json").write_text(
        json.dumps({"classes": list(DIAGNOSIS_CLASSES)}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(registry / "patients.csv", ["person_key", "diagnosis", "include"], people)
    _write_csv(registry / "folds_outer.csv", ["person_key", "outer_fold"], folds)
    config = deepcopy(
        load_gate_config(ROOT / "configs/research/g0_roi_normal_abnormal_gate_b2.yaml")
    )
    config["data"].update(
        {
            "expected_patients": 10,
            "expected_images": 10,
            "expected_normal_patients": 5,
            "expected_abnormal_patients": 5,
        }
    )
    config["evaluation"]["bootstrap_samples"] = 100
    return registry, fold_paths, config


def test_merge_validate_and_evaluate_perfect_fivefold_gate(tmp_path):
    registry, fold_paths, config = _synthetic_contract(tmp_path)
    merged = merge_gate_oof_fold_files(fold_paths, tmp_path / "merged.csv")
    data = load_and_validate_gate_oof(merged, registry, config)
    attention = {
        "contract": {"patients": 10, "unique_images": 10},
        "pooled": {"multi_image_collapse_rate": 0.0},
        "by_fold": {str(fold): {} for fold in range(5)},
    }
    report = evaluate_gate_oof(data, config, attention)

    assert report["contract"]["patients"] == 10
    assert report["metrics"]["roc_auc"] == 1.0
    assert report["metrics"]["macro_f1"] == 1.0
    assert report["roc_auc_95_ci"] == [1.0, 1.0, 1.0]
    assert report["folds_with_roc_auc_at_least_0_75"] == 5
    assert report["performance_attention_gate_passed"] is True


def test_oof_validation_rejects_prediction_not_matching_threshold(tmp_path):
    registry, fold_paths, config = _synthetic_contract(tmp_path)
    columns, rows = None, None
    with fold_paths[0].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    rows[0]["predicted_id"] = 1
    rows[0]["predicted_class"] = "abnormal"
    _write_csv(fold_paths[0], columns, rows)
    merged = merge_gate_oof_fold_files(fold_paths, tmp_path / "merged.csv")

    try:
        load_and_validate_gate_oof(merged, registry, config)
    except ValueError as error:
        assert "probabilities and thresholds" in str(error)
    else:
        raise AssertionError("Mismatched G0 prediction must be rejected")


def test_merge_rejects_missing_or_duplicate_fold(tmp_path):
    _, fold_paths, _ = _synthetic_contract(tmp_path)

    try:
        merge_gate_oof_fold_files(fold_paths[:4], tmp_path / "missing.csv")
    except ValueError as error:
        assert "exactly five" in str(error)
    else:
        raise AssertionError("Missing G0 fold must be rejected")

    try:
        merge_gate_oof_fold_files(
            [fold_paths[0], fold_paths[0], *fold_paths[2:]],
            tmp_path / "duplicate.csv",
        )
    except ValueError as error:
        assert "Duplicate G0 outer fold" in str(error)
    else:
        raise AssertionError("Duplicate G0 fold must be rejected")


def test_attention_must_align_with_oof_patients_and_image_counts(tmp_path):
    registry, fold_paths, config = _synthetic_contract(tmp_path)
    merged = merge_gate_oof_fold_files(fold_paths, tmp_path / "merged.csv")
    data = load_and_validate_gate_oof(merged, registry, config)
    rows = [
        {
            "person_key": key,
            "image_key": f"I{index}",
            "outer_fold": int(data.outer_folds[index]),
            "prediction_level": "image_importance",
            "model_id": data.model_ids[index],
            "image_count": 1,
            "attention_weight": 1.0,
        }
        for index, key in enumerate(data.person_keys)
    ]
    validate_gate_attention_alignment(rows, data)
    audit = audit_attention_rows(rows, collapse_threshold=0.95, max_collapse_rate=0.5)
    assert audit["contract"]["patients"] == 10

    rows.pop()
    try:
        validate_gate_attention_alignment(rows, data)
    except ValueError as error:
        assert "every OOF patient" in str(error)
    else:
        raise AssertionError("Missing G0 attention patient must be rejected")


def test_formal_fold_summaries_validate_mlflow_artifact_and_resource_contract(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = deepcopy(
        load_gate_config(ROOT / "configs/research/g0_roi_normal_abnormal_gate_b2.yaml")
    )
    config_path = project / "g0.yaml"
    config_path.write_text("synthetic-config\n", encoding="utf-8")
    database = project / "tracking" / "mlflow.db"
    database.parent.mkdir()
    run_ids = [f"{index:032x}" for index in range(10)]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE runs (run_uuid TEXT, status TEXT)")
        connection.executemany(
            "INSERT INTO runs VALUES (?, 'FINISHED')",
            [(run_id,) for run_id in run_ids],
        )

    summary_paths = []
    prediction_paths = []
    attention_paths = []
    for fold in range(5):
        prediction = project / "reports" / "oof" / f"G0_fold{fold}.csv"
        _write_csv(
            prediction,
            GATE_OOF_COLUMNS,
            [{column: fold if column == "outer_fold" else "x" for column in GATE_OOF_COLUMNS}],
        )
        attention = project / "reports" / "attention" / f"G0_fold{fold}.csv"
        _write_csv(
            attention,
            ["person_key", "image_key", "outer_fold", "image_count", "attention_weight"],
            [{
                "person_key": f"P{fold}",
                "image_key": f"I{fold}",
                "outer_fold": fold,
                "image_count": 1,
                "attention_weight": 1.0,
            }],
        )
        checkpoint = project / "artifacts" / f"fold{fold}.pt"
        postprocessing = project / "artifacts" / f"fold{fold}.json"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{fold}".encode())
        postprocessing.write_text("{}", encoding="utf-8")
        summary = {
            "experiment_code": "G0",
            "task_type": "binary_normal_abnormal",
            "outer_fold": fold,
            "seed": config["evaluation"]["seeds"][fold],
            "pilot": False,
            "git_dirty": False,
            "git_revision": "a" * 40,
            "status": "EARLY_STOPPED",
            "outer_test_iterated": True,
            "outer_test_used_for_training_or_early_stopping": False,
            "data_fingerprint": config["data_fingerprint"],
            "config_sha256": sha256_file(config_path),
            "pretrained_sha256": config["model"]["pretrained_sha256"],
            "postprocessing": {
                "operating_threshold": {
                    "fit_split": "inner_validation",
                    "constraint_met": True,
                },
                "calibration": {"fit_split": "inner_validation"},
            },
            "mlflow_parent_run_id": run_ids[fold * 2],
            "mlflow_fold_run_id": run_ids[fold * 2 + 1],
            "mlflow_database": database.relative_to(project).as_posix(),
            "prediction_path": prediction.relative_to(project).as_posix(),
            "prediction_sha256": sha256_file(prediction),
            "attention_path": attention.relative_to(project).as_posix(),
            "attention_sha256": sha256_file(attention),
            "best_checkpoint_path": checkpoint.relative_to(project).as_posix(),
            "best_checkpoint_sha256": sha256_file(checkpoint),
            "postprocessing_path": postprocessing.relative_to(project).as_posix(),
            "postprocessing_sha256": sha256_file(postprocessing),
            "elapsed_hours_total": 0.5,
            "peak_gpu_memory_allocated_gb": 1.0,
            "peak_gpu_memory_reserved_gb": 1.2,
            "epochs_completed": 5,
            "gpu": {"gpu_name": "test-gpu", "gpu_memory_total_gb": 10.0},
        }
        summary_path = project / "reports" / f"fold{fold}_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summary_paths.append(summary_path)
        prediction_paths.append(prediction)
        attention_paths.append(attention)

    contract = validate_gate_fold_summaries(
        summary_paths,
        prediction_paths,
        attention_paths,
        config=config,
        config_path=config_path,
        project_root=project,
    )
    assert contract["resource_recording_gate_passed"] is True
    assert contract["git_revision"] == "a" * 40
    assert len(contract["folds"]) == 5

    bad = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    bad["outer_test_used_for_training_or_early_stopping"] = True
    summary_paths[0].write_text(json.dumps(bad), encoding="utf-8")
    try:
        validate_gate_fold_summaries(
            summary_paths,
            prediction_paths,
            attention_paths,
            config=config,
            config_path=config_path,
            project_root=project,
        )
    except ValueError as error:
        assert "no_outer_test_training" in str(error)
    else:
        raise AssertionError("Outer-test leakage in G0 summary must be rejected")
