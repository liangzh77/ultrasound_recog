"""Patient-level semantic finding features and strict outer-fold evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)
from src.research_schema import DIAGNOSIS_CLASSES
from src.research_semantic_contours import SemanticContourRecord


@dataclass(frozen=True)
class SemanticFeatureTable:
    person_keys: tuple[str, ...]
    diagnoses: tuple[str, ...]
    targets: np.ndarray
    outer_folds: np.ndarray
    image_counts: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray


def build_patient_semantic_features(
    image_rows: Sequence[Mapping[str, str]],
    contours: Mapping[str, SemanticContourRecord],
    categories: Sequence[str],
    canvas_pixels: int,
) -> SemanticFeatureTable:
    """Aggregate presence, polygon count and union-area fraction per category."""

    if canvas_pixels <= 0:
        raise ValueError("canvas_pixels must be positive")
    category_order = tuple(categories)
    if not category_order or len(category_order) != len(set(category_order)):
        raise ValueError("Semantic feature categories must be unique and nonempty")
    groups: dict[str, list[Mapping[str, str]]] = {}
    for row in image_rows:
        image_key = str(row["image_key"])
        if image_key not in contours:
            raise ValueError(f"Image lacks semantic contour record: {image_key}")
        groups.setdefault(str(row["person_key"]), []).append(row)
    if not groups:
        raise ValueError("No included patient images were supplied")
    feature_names = tuple(
        f"{category}__{suffix}"
        for category in category_order
        for suffix in ("presence", "object_count", "area_fraction")
    )
    person_keys = tuple(sorted(groups))
    diagnoses: list[str] = []
    targets: list[int] = []
    folds: list[int] = []
    image_counts: list[int] = []
    matrix: list[list[float]] = []
    for person_key in person_keys:
        rows = groups[person_key]
        patient_diagnoses = {str(row["diagnosis"]) for row in rows}
        patient_targets = {int(row["diagnosis_id"]) for row in rows}
        patient_folds = {int(row["outer_fold"]) for row in rows}
        if len(patient_diagnoses) != 1 or len(patient_targets) != 1 or len(patient_folds) != 1:
            raise ValueError(f"Mixed diagnosis or fold for patient {person_key}")
        diagnoses.append(patient_diagnoses.pop())
        targets.append(patient_targets.pop())
        folds.append(patient_folds.pop())
        image_counts.append(len(rows))
        values: list[float] = []
        for category in category_order:
            category_records = [
                contours[str(row["image_key"])].category_masks.get(category)
                for row in rows
            ]
            present = [item for item in category_records if item is not None]
            object_count = sum(item.object_count for item in present)
            positive_pixels = sum(item.positive_pixels for item in present)
            values.extend(
                (
                    float(bool(present)),
                    float(object_count),
                    float(positive_pixels / (len(rows) * canvas_pixels)),
                )
            )
        matrix.append(values)
    return SemanticFeatureTable(
        person_keys=person_keys,
        diagnoses=tuple(diagnoses),
        targets=np.asarray(targets, dtype=np.int64),
        outer_folds=np.asarray(folds, dtype=np.int64),
        image_counts=np.asarray(image_counts, dtype=np.int64),
        feature_names=feature_names,
        features=np.asarray(matrix, dtype=np.float64),
    )


def fit_semantic_feature_oof(
    table: SemanticFeatureTable,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit the fixed scaler and classifier independently inside each outer fold."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class_count = len(DIAGNOSIS_CLASSES)
    probabilities = np.full((len(table.person_keys), class_count), np.nan)
    fold_models: list[dict[str, Any]] = []
    expected_classes = np.arange(class_count)
    for fold in sorted(np.unique(table.outer_folds)):
        test = table.outer_folds == fold
        train = ~test
        if not np.array_equal(np.unique(table.targets[train]), expected_classes):
            raise ValueError(f"M1 training fold {fold} lacks a diagnosis class")
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=seed + int(fold),
                    ),
                ),
            ]
        )
        model.fit(table.features[train], table.targets[train])
        classifier = model.named_steps["classifier"]
        classes = classifier.classes_.astype(int)
        probabilities[np.ix_(test, classes)] = model.predict_proba(table.features[test])
        scaler = model.named_steps["scaler"]
        fold_models.append(
            {
                "outer_fold": int(fold),
                "classes": classes.tolist(),
                "feature_names": list(table.feature_names),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coef": classifier.coef_.tolist(),
                "intercept": classifier.intercept_.tolist(),
                "n_iter": classifier.n_iter_.astype(int).tolist(),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("M1 OOF probabilities are incomplete")
    return probabilities, fold_models


def evaluate_m1_against_m0(
    table: SemanticFeatureTable,
    m1_probabilities: np.ndarray,
    m0_probabilities: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if m0_probabilities.shape != m1_probabilities.shape:
        raise ValueError("M0 and M1 probability shapes differ")
    m0_metrics = compute_patient_metrics(
        table.targets, m0_probabilities, DIAGNOSIS_CLASSES
    )
    m1_metrics = compute_patient_metrics(
        table.targets, m1_probabilities, DIAGNOSIS_CLASSES
    )
    paired = paired_bootstrap_macro_f1(
        table.targets,
        m0_probabilities,
        m1_probabilities,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    fold_metrics: dict[str, Any] = {}
    for fold in sorted(np.unique(table.outer_folds)):
        selected = table.outer_folds == fold
        first = compute_patient_metrics(
            table.targets[selected], m0_probabilities[selected], DIAGNOSIS_CLASSES
        )
        second = compute_patient_metrics(
            table.targets[selected], m1_probabilities[selected], DIAGNOSIS_CLASSES
        )
        fold_metrics[str(fold)] = {
            "M0_macro_f1": first["macro_f1"],
            "M1_macro_f1": second["macro_f1"],
            "M1_minus_M0": second["macro_f1"] - first["macro_f1"],
        }
    return {
        "M0": {
            "metrics": m0_metrics,
            "macro_f1_ci95": list(
                bootstrap_macro_f1_ci(
                    table.targets,
                    m0_probabilities,
                    n_bootstrap=bootstrap_samples,
                    seed=bootstrap_seed,
                )
            ),
        },
        "M1": {
            "metrics": m1_metrics,
            "macro_f1_ci95": list(
                bootstrap_macro_f1_ci(
                    table.targets,
                    m1_probabilities,
                    n_bootstrap=bootstrap_samples,
                    seed=bootstrap_seed,
                )
            ),
        },
        "M1_minus_M0": paired,
        "fold_metrics": fold_metrics,
    }
