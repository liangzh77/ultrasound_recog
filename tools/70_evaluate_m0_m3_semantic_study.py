"""Evaluate frozen M0--M3 OOF comparisons and one fixed M2/C3 fusion."""

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

from src.common_paths import PATIENT_MULTIMODAL_EXPERIMENT_DIR  # noqa: E402
from src.research_oof import PROBABILITY_COLUMNS  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402
from src.research_semantic_contours import sha256_file  # noqa: E402
from src.research_semantic_study import (  # noqa: E402
    MSeriesData,
    evaluate_m_series,
    fuse_image_and_clinical,
    hierarchical_image_c3_comparator,
    validate_probability_matrix,
)
from src.research_tracking import LocalResearchTracker  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/m3_semantic_c3_fixed_fusion.yaml"
CLINICAL_COLUMNS = ("prob_ra", "prob_ga", "prob_spa", "prob_oa", "prob_injury")


def _resolve(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("M3 paths must be project-relative")
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty M-series input: {path.name}")
    return rows


def _index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = row.get("person_key", "")
        if not key or key in result:
            raise ValueError(f"{label} has an empty or duplicate person_key")
        result[key] = row
    return result


def _require_clean_git() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Formal M3 evaluation requires a clean Git worktree")
    return revision


def _load(config: dict) -> MSeriesData:
    specs = config["inputs"]
    rows_by_model = {}
    for label, spec in specs.items():
        path = _resolve(spec["path"])
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"{label} input SHA-256 mismatch")
        rows_by_model[label] = _index(_read_csv(path), label)
    m2_index = rows_by_model["M2"]
    expected_patients = int(config["cohort"]["expected_patients"])
    if len(m2_index) != expected_patients:
        raise ValueError("M2 patient count differs from M3 cohort")
    for label in ("M0", "M1"):
        if set(rows_by_model[label]) != set(m2_index):
            raise ValueError(f"{label} patient set differs from M2")
    c3_index = rows_by_model["C3"]
    if len(c3_index) != int(config["cohort"]["expected_clinical_rows"]):
        raise ValueError("C3 row count differs from M3 cohort")
    if not set(c3_index) < set(m2_index):
        raise ValueError("C3 keys must be a strict subset of the full cohort")

    person_keys = tuple(sorted(m2_index))
    references = []
    targets = []
    folds = []
    image_counts = []
    matrices = {label: [] for label in ("M0", "M1", "M2")}
    c3 = np.full((len(person_keys), 5), np.nan, dtype=np.float64)
    available = np.zeros(len(person_keys), dtype=bool)
    for index, key in enumerate(person_keys):
        anchor = m2_index[key]
        target = int(anchor["reference_id"])
        fold = int(anchor["outer_fold"])
        reference = anchor["reference_class"]
        targets.append(target)
        folds.append(fold)
        references.append(reference)
        image_counts.append(int(anchor["image_count"]))
        for label in ("M0", "M1", "M2"):
            row = rows_by_model[label][key]
            if (
                int(row["reference_id"]) != target
                or int(row["outer_fold"]) != fold
                or row["reference_class"] != reference
            ):
                raise ValueError(f"{label} fold/reference contract differs from M2")
            matrices[label].append([float(row[column]) for column in PROBABILITY_COLUMNS])
        clinical = c3_index.get(key)
        if clinical is not None:
            if (
                target == 0
                or int(clinical["reference_id"]) != target - 1
                or int(clinical["outer_fold"]) != fold
                or clinical["reference_class"] != reference
            ):
                raise ValueError("C3 fold/reference contract differs from M2")
            available[index] = True
            c3[index] = [float(clinical[column]) for column in CLINICAL_COLUMNS]
    if set(folds) != set(config["cohort"]["expected_outer_folds"]):
        raise ValueError("M3 outer folds differ from frozen contract")
    arrays = {
        label: validate_probability_matrix(np.asarray(values), 6)
        for label, values in matrices.items()
    }
    fusion = config["fixed_fusion"]
    m3 = fuse_image_and_clinical(
        arrays["M2"],
        c3,
        available,
        float(fusion["clinical_weight_within_abnormal"]),
        float(fusion["semantic_image_weight_within_abnormal"]),
    )
    m0_c3 = hierarchical_image_c3_comparator(arrays["M0"], c3, available)
    return MSeriesData(
        person_keys=person_keys,
        references=tuple(references),
        targets=np.asarray(targets, dtype=np.int64),
        outer_folds=np.asarray(folds, dtype=np.int64),
        image_counts=np.asarray(image_counts, dtype=np.int64),
        clinical_available=available,
        m0=arrays["M0"],
        m1=arrays["M1"],
        m2=arrays["M2"],
        c3=c3,
        m0_c3=m0_c3,
        m3=m3,
    )


def main() -> int:
    started = time.perf_counter()
    below_normal_priority = set_below_normal_priority()
    revision = _require_clean_git()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_after_m2_before_m3_results":
        raise ValueError("M3 method is not frozen")
    fusion = config["fixed_fusion"]
    if fusion.get("search_weights") is not False:
        raise ValueError("M3 forbids weight search")
    data = _load(config)
    evaluation = evaluate_m_series(
        data,
        int(config["evaluation"]["bootstrap_samples"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    output_path = _resolve(config["outputs"]["oof_csv"])
    report_path = _resolve(config["outputs"]["result_json"])
    if output_path.exists() or report_path.exists():
        raise FileExistsError("M3 formal outputs already exist and will not be overwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, key in enumerate(data.person_keys):
        ranking = data.m3[index].argsort()[::-1]
        row = {
            "prediction_level": "patient",
            "person_key": key,
            "outer_fold": int(data.outer_folds[index]),
            "reference_class": data.references[index],
            "reference_id": int(data.targets[index]),
            "image_count": int(data.image_counts[index]),
            "clinical_available": int(data.clinical_available[index]),
            "model_id": "M3-fixed-M2-C3-fusion",
            "top1": DIAGNOSIS_CLASSES[int(ranking[0])],
            "top2": DIAGNOSIS_CLASSES[int(ranking[1])],
        }
        for prefix, probabilities in (
            ("m3", data.m3[index]),
            ("m0_c3", data.m0_c3[index]),
        ):
            row.update(
                {
                    f"{prefix}_{column}": float(probabilities[class_id])
                    for class_id, column in enumerate(PROBABILITY_COLUMNS)
                }
            )
        rows.append(row)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking", "patient-primary-diagnosis"
    )
    with tracker.parent_run(
        "M3-semantic-C3-fixed-fusion-formal",
        {
            "study_code": "M3",
            "git_revision": revision,
            "config_sha256": sha256_file(CONFIG_PATH),
            "clinical_weight": fusion["clinical_weight_within_abnormal"],
            "semantic_image_weight": fusion["semantic_image_weight_within_abnormal"],
            "weight_search": False,
            "allow_2026": False,
        },
    ) as run:
        mlflow_parent_run_id = run.info.run_id
        tracker.log_metrics(
            {
                f"{name}_macro_f1": report["metrics"]["macro_f1"]
                for name, report in evaluation["models"].items()
            }
        )
        tracker.log_metrics(
            {
                name: report["observed_delta"]
                for name, report in evaluation["comparisons"].items()
            }
        )
    report = {
        "status": "completed_frozen_m0_m3_retrospective_evaluation",
        "study_code": "M0-M3",
        "git_commit": revision,
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_sha256": {
            label: spec["sha256"] for label, spec in config["inputs"].items()
        },
        "fixed_fusion": fusion,
        "evaluation": evaluation,
        "output_oof": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output_path),
        },
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
        "interpretation_contract": config["interpretation_contract"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "macro_f1": {
                    name: values["metrics"]["macro_f1"]
                    for name, values in evaluation["models"].items()
                },
                "deltas": {
                    name: values["observed_delta"]
                    for name, values in evaluation["comparisons"].items()
                },
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
