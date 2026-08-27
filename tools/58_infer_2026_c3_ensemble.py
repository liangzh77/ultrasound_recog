"""Run the frozen five-member C3 ensemble on the diagnosis-hidden 2026 cohort."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_auto_roi import require_sha256, sha256_file  # noqa: E402
from src.research_clinical import (  # noqa: E402
    CENSOR_FEATURES,
    CLINICAL_CLASSES,
    CLINICAL_PROBABILITY_COLUMNS,
    MISSING_FEATURES,
    clinical_feature_matrix,
    load_clinical_config,
)


R2_CONFIG_PATH = ROOT / "configs/research/r2_2026_frozen_inference.yaml"
C3_CONFIG_PATH = ROOT / "configs/research/c0_c4_clinical_logreg.yaml"
PRIVATE_DIR = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "r2_2026"
FEATURE_PATH = PRIVATE_DIR / "2026_clinical_features.csv"
FEATURE_AUDIT_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026" / "2026_clinical_feature_audit.json"
OUTPUT_PATH = PRIVATE_DIR / "2026_c3_conditional_probabilities.csv"
REPORT_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026" / "2026_c3_component_manifest.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _validate_feature_rows(
    rows: list[dict[str, str]], expected_features: tuple[str, ...], expected_patients: int
) -> None:
    expected_columns = (
        "person_key",
        *expected_features,
        *MISSING_FEATURES,
        *CENSOR_FEATURES,
    )
    if not rows or tuple(rows[0]) != expected_columns:
        raise ValueError("2026 clinical feature columns differ from the frozen C3 order")
    keys = [row["person_key"] for row in rows]
    if len(rows) != expected_patients or len(keys) != len(set(keys)):
        raise ValueError("2026 clinical features must cover each blind patient exactly once")
    if any(not key.startswith("KNEE_R2_") for key in keys):
        raise ValueError("2026 clinical features contain an unexpected patient key")


def _load_member(
    definition: dict[str, str],
    expected_features: tuple[str, ...],
) -> tuple[Any, dict[str, str]]:
    path = (ROOT / definition["path"]).resolve()
    digest = require_sha256(path, definition["sha256"], "frozen C3 model")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or set(artifact) != {"pipeline", "features", "classes"}:
        raise ValueError("frozen C3 artifact structure is invalid")
    if tuple(artifact["features"]) != expected_features:
        raise ValueError("frozen C3 artifact feature order is invalid")
    if tuple(artifact["classes"]) != CLINICAL_CLASSES:
        raise ValueError("frozen C3 artifact class order is invalid")
    pipeline = artifact["pipeline"]
    classifier_classes = tuple(int(value) for value in pipeline.named_steps["classifier"].classes_)
    if classifier_classes != tuple(range(len(CLINICAL_CLASSES))):
        raise ValueError("frozen C3 classifier probability order is invalid")
    return pipeline, {"path": definition["path"], "sha256": digest}


def _write_predictions(
    path: Path, rows: list[dict[str, str]], probabilities: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("person_key", *CLINICAL_PROBABILITY_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, vector in zip(rows, probabilities, strict=True):
            writer.writerow(
                {
                    "person_key": row["person_key"],
                    **{
                        column: format(float(value), ".17g")
                        for column, value in zip(
                            CLINICAL_PROBABILITY_COLUMNS, vector, strict=True
                        )
                    },
                }
            )


def main() -> int:
    revision, dirty = _git_revision()
    if dirty:
        raise RuntimeError("Frozen 2026 C3 inference requires a clean Git worktree")

    r2_config = yaml.safe_load(R2_CONFIG_PATH.read_text(encoding="utf-8"))
    c3_config = load_clinical_config(C3_CONFIG_PATH)
    clinical_contract = r2_config["clinical_ensemble"]
    expected_features = tuple(clinical_contract["features"])
    if expected_features != tuple(c3_config["experiments"]["C3"]["features"]):
        raise ValueError("R2 and frozen C3 feature contracts differ")

    audit = json.loads(FEATURE_AUDIT_PATH.read_text(encoding="utf-8"))
    feature_sha = require_sha256(
        FEATURE_PATH,
        audit["private_feature_table_sha256"],
        "private 2026 C3 feature table",
    )
    if audit["source_workbook_sha256"] != r2_config["cohort"]["clinical_workbook_sha256"]:
        raise ValueError("2026 feature audit workbook hash differs from the R2 contract")
    if audit.get("reference_diagnoses_read") is not False:
        raise ValueError("2026 feature audit does not preserve diagnosis blinding")

    rows = _read_csv(FEATURE_PATH)
    _validate_feature_rows(
        rows, expected_features, int(r2_config["cohort"]["patients"])
    )
    matrix = clinical_feature_matrix(
        rows,
        expected_features,
        float(c3_config["feature_contract"]["hla_b27_missing_value"]),
    )

    model_definitions = clinical_contract["models"]
    if {int(fold) for fold in model_definitions} != set(range(5)):
        raise ValueError("R2 requires exactly five frozen C3 members")
    member_probabilities = []
    model_manifest = []
    for fold in range(5):
        definition = model_definitions.get(fold, model_definitions.get(str(fold)))
        pipeline, model_record = _load_member(definition, expected_features)
        probabilities = np.asarray(pipeline.predict_proba(matrix), dtype=np.float64)
        if probabilities.shape != (len(rows), len(CLINICAL_CLASSES)):
            raise ValueError("frozen C3 member returned an invalid probability shape")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("frozen C3 member returned invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("frozen C3 member probabilities do not sum to one")
        member_probabilities.append(probabilities)
        model_manifest.append({"fold": fold, **model_record})

    ensemble = np.mean(np.stack(member_probabilities, axis=0), axis=0)
    if not np.allclose(ensemble.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("C3 ensemble probabilities do not sum to one")
    _write_predictions(OUTPUT_PATH, rows, ensemble)

    report = {
        "experiment_code": "R2-C3-component",
        "status": "blind_c3_component_frozen",
        "patients": len(rows),
        "classes": list(CLINICAL_CLASSES),
        "aggregation": "equal_mean_of_five_conditional_probability_vectors",
        "feature_table_sha256": feature_sha,
        "model_manifest": model_manifest,
        "private_prediction_sha256": sha256_file(OUTPUT_PATH),
        "git_commit": revision,
        "git_dirty_at_inference": False,
        "reference_diagnoses_read": False,
        "patient_probabilities_publicly_reported": False,
        "formal_six_class_prediction_generated": False,
        "route_selection_affected": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(REPORT_PATH), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
