"""Run the frozen M1 semantic finding feature fivefold evaluation."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_EXPERIMENT_DIR,
    PATIENT_MULTIMODAL_REGISTRY_DIR,
)
from src.research_runtime import set_below_normal_priority  # noqa: E402
from src.research_oof import PROBABILITY_COLUMNS  # noqa: E402
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402
from src.research_semantic_contours import (  # noqa: E402
    load_semantic_contour_manifest,
    sha256_file,
)
from src.research_semantic_features import (  # noqa: E402
    build_patient_semantic_features,
    evaluate_m1_against_m0,
    fit_semantic_feature_oof,
)
from src.research_tracking import LocalResearchTracker  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/m1_semantic_finding_features_logreg.yaml"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path.name}")
    return rows


def _resolve(relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ValueError("M1 paths must be project-relative")
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _require_clean_git() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Formal M1 evaluation requires a clean Git worktree")
    return revision


def _load_m0(table, spec: dict) -> np.ndarray:
    path = _resolve(spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError("M0 E1 OOF SHA-256 mismatch")
    index = {row["person_key"]: row for row in _read_csv(path)}
    if len(index) != len(table.person_keys) or set(index) != set(table.person_keys):
        raise ValueError("M0 E1 patient set differs from M1")
    matrix = []
    for position, person_key in enumerate(table.person_keys):
        row = index[person_key]
        if (
            int(row["outer_fold"]) != int(table.outer_folds[position])
            or int(row["reference_id"]) != int(table.targets[position])
        ):
            raise ValueError("M0 E1 fold/reference contract differs from M1")
        matrix.append([float(row[column]) for column in PROBABILITY_COLUMNS])
    result = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("M0 probabilities do not sum to one")
    return result


def main() -> int:
    started = time.perf_counter()
    below_normal_priority = set_below_normal_priority()
    git_commit = _require_clean_git()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_after_semantic_preparation_before_m1_results":
        raise ValueError("M1 method is not frozen")
    categories = tuple(config["finding_categories"])
    images = [
        row
        for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv")
        if row["include"] == "1"
    ]
    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_outer.csv")
    }
    image_rows = [{**row, "outer_fold": str(folds[row["person_key"]])} for row in images]
    manifest_spec = config["inputs"]["semantic_manifest"]
    manifest_path = _resolve(manifest_spec["path"])
    contours = load_semantic_contour_manifest(
        manifest_path,
        ROOT,
        manifest_spec["sha256"],
        {row["image_key"] for row in image_rows},
        categories,
        verify_mask_hashes=True,
    )
    table = build_patient_semantic_features(
        image_rows,
        contours,
        categories,
        int(config["feature_contract"]["canvas_pixels_per_image"]),
    )
    if len(table.person_keys) != int(config["cohort"]["expected_patients"]):
        raise ValueError("M1 patient count differs from frozen contract")
    if len(images) != int(config["cohort"]["expected_images"]):
        raise ValueError("M1 image count differs from frozen contract")
    if sorted(np.unique(table.outer_folds).tolist()) != config["cohort"]["expected_outer_folds"]:
        raise ValueError("M1 outer folds differ from frozen contract")
    m1, fold_models = fit_semantic_feature_oof(table, int(config["seed"]))
    m0 = _load_m0(table, config["inputs"]["m0_e1_oof"])
    evaluation = evaluate_m1_against_m0(
        table,
        m1,
        m0,
        int(config["evaluation"]["bootstrap_samples"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )

    outputs = {name: _resolve(path) for name, path in config["outputs"].items()}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("M1 formal outputs already exist and will not be overwritten")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    oof_rows = []
    for index, person_key in enumerate(table.person_keys):
        feature_row = {
            "person_key": person_key,
            "outer_fold": int(table.outer_folds[index]),
            "reference_class": table.diagnoses[index],
            "reference_id": int(table.targets[index]),
            "image_count": int(table.image_counts[index]),
        }
        feature_row.update(
            {
                name: float(table.features[index, feature_index])
                for feature_index, name in enumerate(table.feature_names)
            }
        )
        feature_rows.append(feature_row)
        ranking = m1[index].argsort()[::-1]
        oof_row = {
            "prediction_level": "patient",
            "person_key": person_key,
            "outer_fold": int(table.outer_folds[index]),
            "reference_class": table.diagnoses[index],
            "reference_id": int(table.targets[index]),
            "image_count": int(table.image_counts[index]),
            "model_id": "M1-fixed-semantic-logreg",
            "top1": DIAGNOSIS_CLASSES[int(ranking[0])],
            "top2": DIAGNOSIS_CLASSES[int(ranking[1])],
        }
        oof_row.update(
            {
                column: float(m1[index, class_id])
                for class_id, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        oof_rows.append(oof_row)
    for path, rows in (
        (outputs["feature_csv"], feature_rows),
        (outputs["oof_csv"], oof_rows),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    outputs["weights_json"].write_text(
        json.dumps(fold_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking",
        "patient-primary-diagnosis",
    )
    with tracker.parent_run(
        "M1-semantic-finding-features-formal",
        {
            "study_code": "M1",
            "prediction_level": "patient",
            "data_fingerprint": config["data_fingerprint"],
            "config_sha256": sha256_file(CONFIG_PATH),
            "git_revision": git_commit,
            "folds": "0,1,2,3,4",
            "features": len(table.feature_names),
            "rgb_used": False,
            "excel_used": False,
        },
    ) as run:
        mlflow_parent_run_id = run.info.run_id
        tracker.log_metrics(
            {
                "M0_macro_f1": evaluation["M0"]["metrics"]["macro_f1"],
                "M1_macro_f1": evaluation["M1"]["metrics"]["macro_f1"],
                "M1_minus_M0": evaluation["M1_minus_M0"]["observed_delta"],
                "M1_ece": evaluation["M1"]["metrics"]["ece"],
            }
        )
    report = {
        "status": "completed_retrospective_semantic_feature_evaluation",
        "study_code": "M1",
        "git_commit": git_commit,
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_sha256": {
            "semantic_manifest": manifest_spec["sha256"],
            "M0_E1_oof": config["inputs"]["m0_e1_oof"]["sha256"],
        },
        "patients": len(table.person_keys),
        "images": len(images),
        "features": len(table.feature_names),
        "finding_categories": list(categories),
        "evaluation": evaluation,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "numeric_thread_limit": 2,
            "below_normal_priority": below_normal_priority,
            "gpu_used": False,
        },
        "tracking": {
            "mlflow_experiment": "patient-primary-diagnosis",
            "mlflow_parent_run_id": mlflow_parent_run_id,
        },
        "outputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
            if name != "result_json"
        },
        "interpretation_contract": config["interpretation_contract"],
    }
    outputs["result_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "M0_macro_f1": evaluation["M0"]["metrics"]["macro_f1"],
                "M1_macro_f1": evaluation["M1"]["metrics"]["macro_f1"],
                "M1_minus_M0": evaluation["M1_minus_M0"]["observed_delta"],
                "result_sha256": sha256_file(outputs["result_json"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
