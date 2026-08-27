"""Evaluate the preregistered three-feature L0 mask-presence/area proxy."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_metrics import (  # noqa: E402
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)
from src.research_oof import (  # noqa: E402
    PROBABILITY_COLUMNS,
    evaluate_oof_file,
)
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/l0_mask_only_proxy_audit.yaml"
OUTPUT_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "lesion_guided"
PROXY_OOF_PATH = OUTPUT_DIR / "L0_mask_only_proxy_oof.csv"
REPORT_PATH = OUTPUT_DIR / "L0_mask_only_proxy_audit.json"
L0_OOF_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "oof/L0_oof.csv"
E1_OOF_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "oof/E1_oof.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_patient_features(
    image_rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
    mask_pixels: int,
) -> dict[str, dict[str, float | int]]:
    included = [row for row in image_rows if row["include"] == "1"]
    image_to_person = {row["image_key"]: row["person_key"] for row in included}
    if len(image_to_person) != len(included):
        raise ValueError("Included image keys must be unique")
    mask_by_image = {row["image_key"]: row for row in mask_rows}
    if len(mask_by_image) != len(mask_rows) or set(mask_by_image) != set(image_to_person):
        raise ValueError("Mask manifest must exactly match included images")
    areas: dict[str, list[float]] = defaultdict(list)
    for image_key, person_key in image_to_person.items():
        positive_pixels = int(mask_by_image[image_key]["positive_pixels"])
        if not 0 <= positive_pixels <= mask_pixels:
            raise ValueError(f"Invalid positive pixel count for {image_key}")
        areas[person_key].append(positive_pixels / mask_pixels)
    features = {}
    for person_key, values in areas.items():
        array = np.asarray(values, dtype=np.float64)
        features[person_key] = {
            "positive_mask_image_fraction": float(np.mean(array > 0)),
            "mean_mask_area_fraction_all_images": float(np.mean(array)),
            "maximum_mask_area_fraction": float(np.max(array)),
            "image_count": int(len(array)),
            "all_zero_mask": int(not np.any(array > 0)),
        }
    return features


def fit_proxy_oof(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probabilities = np.zeros((len(targets), len(DIAGNOSIS_CLASSES)), dtype=np.float64)
    models = []
    for fold in config["evaluation"]["outer_folds"]:
        train = folds != fold
        test = folds == fold
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config["model"]["C"]),
                class_weight=config["model"]["class_weight"],
                solver=config["model"]["solver"],
                max_iter=int(config["model"]["max_iter"]),
                random_state=int(config["model"]["random_state"]),
            ),
        )
        pipeline.fit(features[train], targets[train])
        classifier = pipeline.named_steps["logisticregression"]
        if classifier.classes_.tolist() != list(range(len(DIAGNOSIS_CLASSES))):
            raise ValueError("Every proxy training fold must contain all six classes")
        probabilities[test] = pipeline.predict_proba(features[test])
        scaler = pipeline.named_steps["standardscaler"]
        models.append(
            {
                "outer_fold": int(fold),
                "train_patients": int(np.sum(train)),
                "test_patients": int(np.sum(test)),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "classifier_intercept": classifier.intercept_.tolist(),
                "classifier_coefficients": classifier.coef_.tolist(),
                "iterations": classifier.n_iter_.tolist(),
            }
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Proxy probabilities are invalid")
    return probabilities, models


def load_oof_probabilities(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    return {
        row["person_key"]: np.asarray(
            [float(row[column]) for column in PROBABILITY_COLUMNS], dtype=np.float64
        )
        for row in rows
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_method_preregistered_before_l0_classification_results":
        raise ValueError("Mask-only proxy method must be frozen")
    manifest_path = ROOT / config["mask_manifest_path"]
    if sha256_file(manifest_path) != config["mask_manifest_sha256"]:
        raise ValueError("Mask manifest SHA-256 mismatch")
    patient_features = build_patient_features(
        read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv"),
        read_csv(manifest_path),
        int(config["mask_pixels"]),
    )
    patients = {
        row["person_key"]: row
        for row in read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "patients.csv")
        if row["include"] == "1"
    }
    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_outer.csv")
    }
    if set(patient_features) != set(patients) or set(patients) != set(folds):
        raise ValueError("Patient features, labels, and folds must match exactly")
    person_keys = sorted(patients)
    feature_names = list(config["features"])
    matrix = np.asarray(
        [[patient_features[key][name] for name in feature_names] for key in person_keys],
        dtype=np.float64,
    )
    targets = np.asarray(
        [int(patients[key]["diagnosis_id"]) for key in person_keys], dtype=np.int64
    )
    outer_folds = np.asarray([folds[key] for key in person_keys], dtype=np.int64)
    probabilities, models = fit_proxy_oof(matrix, targets, outer_folds, config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROXY_OOF_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "prediction_level",
            "person_key",
            "outer_fold",
            "reference_class",
            "reference_id",
            *PROBABILITY_COLUMNS,
            "image_count",
            "model_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, person_key in enumerate(person_keys):
            target = int(targets[index])
            writer.writerow(
                {
                    "prediction_level": "patient",
                    "person_key": person_key,
                    "outer_fold": int(outer_folds[index]),
                    "reference_class": DIAGNOSIS_CLASSES[target],
                    "reference_id": target,
                    **dict(
                        zip(
                            PROBABILITY_COLUMNS,
                            probabilities[index].tolist(),
                            strict=True,
                        )
                    ),
                    "image_count": patient_features[person_key]["image_count"],
                    "model_id": f"L0-mask-only-proxy-fold{outer_folds[index]}",
                }
            )

    proxy_evaluation = evaluate_oof_file(
        PROXY_OOF_PATH,
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        n_bootstrap=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    l0 = load_oof_probabilities(L0_OOF_PATH)
    e1 = load_oof_probabilities(E1_OOF_PATH)
    if set(l0) != set(person_keys) or set(e1) != set(person_keys):
        raise ValueError("L0/E1 OOF patient sets differ from proxy patients")
    subgroup_results = {}
    for name, wanted_zero in (("all_zero_mask", 1), ("any_positive_mask", 0)):
        indices = np.asarray(
            [
                index
                for index, key in enumerate(person_keys)
                if patient_features[key]["all_zero_mask"] == wanted_zero
            ],
            dtype=np.int64,
        )
        subgroup_targets = targets[indices]
        l0_probabilities = np.asarray([l0[person_keys[index]] for index in indices])
        e1_probabilities = np.asarray([e1[person_keys[index]] for index in indices])
        subgroup_results[name] = {
            "patients": int(len(indices)),
            "class_support": np.bincount(
                subgroup_targets, minlength=len(DIAGNOSIS_CLASSES)
            ).tolist(),
            "l0_metrics": compute_patient_metrics(
                subgroup_targets, l0_probabilities, DIAGNOSIS_CLASSES
            ),
            "e1_metrics": compute_patient_metrics(
                subgroup_targets, e1_probabilities, DIAGNOSIS_CLASSES
            ),
            "paired_l0_minus_e1": paired_bootstrap_macro_f1(
                subgroup_targets,
                e1_probabilities,
                l0_probabilities,
                n_bootstrap=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["evaluation"]["bootstrap_seed"]),
            ),
        }
    report = {
        "status": "completed_descriptive_bias_proxy_not_model_selection",
        "experiment_code": "L0-mask-only-proxy",
        "data_fingerprint": config["data_fingerprint"],
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "mask_manifest_sha256": sha256_file(manifest_path),
        "features": feature_names,
        "rgb_used": False,
        "category_names_used": False,
        "tuning_performed": False,
        "patients": len(person_keys),
        "all_zero_mask_patients": sum(
            patient_features[key]["all_zero_mask"] for key in person_keys
        ),
        "proxy_oof_path": PROXY_OOF_PATH.relative_to(ROOT).as_posix(),
        "proxy_oof_sha256": sha256_file(PROXY_OOF_PATH),
        "proxy_evaluation": proxy_evaluation,
        "fold_models": models,
        "l0_e1_subgroups": subgroup_results,
        "contains_raw_paths": False,
        "allow_2026": False,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "report_sha256": sha256_file(REPORT_PATH),
                "proxy_macro_f1": proxy_evaluation["metrics"]["macro_f1"],
                "all_zero_mask_patients": report["all_zero_mask_patients"],
                "subgroups": {
                    name: {
                        "patients": value["patients"],
                        "l0_macro_f1": value["l0_metrics"]["macro_f1"],
                        "e1_macro_f1": value["e1_metrics"]["macro_f1"],
                        "delta": value["paired_l0_minus_e1"]["observed_delta"],
                    }
                    for name, value in subgroup_results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

