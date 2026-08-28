"""Frozen M0--M3 comparison and workflow-aware M2/C3 fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.research_clinical import CLINICAL_CLASSES
from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)
from src.research_schema import DIAGNOSIS_CLASSES


@dataclass(frozen=True)
class MSeriesData:
    person_keys: tuple[str, ...]
    references: tuple[str, ...]
    targets: np.ndarray
    outer_folds: np.ndarray
    image_counts: np.ndarray
    clinical_available: np.ndarray
    m0: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    c3: np.ndarray
    m0_c3: np.ndarray
    m3: np.ndarray


def validate_probability_matrix(matrix: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != classes:
        raise ValueError("Probability matrix has an invalid shape")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Probability matrix contains invalid values")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to one")
    return matrix


def fuse_image_and_clinical(
    image_probabilities: np.ndarray,
    clinical_probabilities: np.ndarray,
    clinical_available: np.ndarray,
    clinical_weight: float,
    image_weight: float,
) -> np.ndarray:
    """Keep image-only rows; blend abnormal conditionals where C3 exists."""

    image = validate_probability_matrix(image_probabilities, 6)
    clinical = np.asarray(clinical_probabilities, dtype=np.float64)
    available = np.asarray(clinical_available, dtype=bool)
    if clinical.shape != (len(image), 5) or available.shape != (len(image),):
        raise ValueError("Clinical probabilities/availability differ from image rows")
    if not np.isclose(clinical_weight + image_weight, 1.0):
        raise ValueError("Fusion weights must sum to one")
    if clinical_weight < 0 or image_weight < 0:
        raise ValueError("Fusion weights must be nonnegative")
    if available.any():
        validate_probability_matrix(clinical[available], 5)
    result = image.copy()
    if not available.any():
        return result
    normal = image[available, :1]
    abnormal = image[available, 1:]
    denominator = abnormal.sum(axis=1, keepdims=True)
    if (denominator <= 1e-8).any():
        raise ValueError("Image abnormal conditional denominator is invalid")
    image_conditionals = abnormal / denominator
    blended = clinical_weight * clinical[available] + image_weight * image_conditionals
    result[available, :1] = normal
    result[available, 1:] = (1.0 - normal) * blended
    return validate_probability_matrix(result, 6)


def hierarchical_image_c3_comparator(
    image_probabilities: np.ndarray,
    clinical_probabilities: np.ndarray,
    clinical_available: np.ndarray,
) -> np.ndarray:
    """Apply the existing image normal gate plus C3 abnormal condition where available."""

    return fuse_image_and_clinical(
        image_probabilities,
        clinical_probabilities,
        clinical_available,
        clinical_weight=1.0,
        image_weight=0.0,
    )


def _model_report(
    targets: np.ndarray,
    folds: np.ndarray,
    probabilities: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "metrics": compute_patient_metrics(targets, probabilities, DIAGNOSIS_CLASSES),
        "macro_f1_ci95": list(
            bootstrap_macro_f1_ci(
                targets,
                probabilities,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed,
            )
        ),
        "fold_macro_f1": {
            str(fold): compute_patient_metrics(
                targets[folds == fold], probabilities[folds == fold], DIAGNOSIS_CLASSES
            )["macro_f1"]
            for fold in sorted(np.unique(folds))
        },
    }


def _comparison(
    targets: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired = paired_bootstrap_macro_f1(
        targets,
        baseline,
        candidate,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    fold_deltas = {}
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        first = compute_patient_metrics(
            targets[selected], baseline[selected], DIAGNOSIS_CLASSES
        )["macro_f1"]
        second = compute_patient_metrics(
            targets[selected], candidate[selected], DIAGNOSIS_CLASSES
        )["macro_f1"]
        fold_deltas[str(fold)] = second - first
    baseline_metrics = compute_patient_metrics(targets, baseline, DIAGNOSIS_CLASSES)
    candidate_metrics = compute_patient_metrics(targets, candidate, DIAGNOSIS_CLASSES)
    class_deltas = {
        left["class_name"]: right["f1"] - left["f1"]
        for left, right in zip(
            baseline_metrics["per_class"], candidate_metrics["per_class"], strict=True
        )
    }
    return {
        **paired,
        "fold_macro_f1_deltas": fold_deltas,
        "positive_folds": sum(value > 0 for value in fold_deltas.values()),
        "per_class_f1_deltas": class_deltas,
    }


def evaluate_m_series(
    data: MSeriesData,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    models = {
        "M0": data.m0,
        "M1": data.m1,
        "M2": data.m2,
        "M0_C3_hierarchical": data.m0_c3,
        "M3": data.m3,
    }
    reports = {
        name: _model_report(
            data.targets,
            data.outer_folds,
            probabilities,
            bootstrap_samples,
            bootstrap_seed,
        )
        for name, probabilities in models.items()
    }
    comparisons = {
        "M1_minus_M0": (data.m0, data.m1),
        "M2_minus_M0": (data.m0, data.m2),
        "M2_minus_M1": (data.m1, data.m2),
        "M3_minus_M2": (data.m2, data.m3),
        "M3_minus_M0_C3_hierarchical": (data.m0_c3, data.m3),
    }
    comparison_reports = {
        name: _comparison(
            data.targets,
            data.outer_folds,
            baseline,
            candidate,
            bootstrap_samples,
            bootstrap_seed,
        )
        for name, (baseline, candidate) in comparisons.items()
    }
    selected = data.clinical_available
    abnormal_targets = data.targets[selected] - 1
    # Boolean indexing currently returns a copy, but make the non-mutation
    # contract explicit: evaluation must never alter frozen input OOF arrays.
    m2_conditional = data.m2[selected, 1:].copy()
    m2_conditional /= m2_conditional.sum(axis=1, keepdims=True)
    m3_conditional = data.m3[selected, 1:].copy()
    m3_conditional /= m3_conditional.sum(axis=1, keepdims=True)
    return {
        "patients": len(data.person_keys),
        "clinical_available_patients": int(selected.sum()),
        "models": reports,
        "comparisons": comparison_reports,
        "abnormal_condition_analysis": {
            "M2": compute_patient_metrics(
                abnormal_targets, m2_conditional, CLINICAL_CLASSES
            ),
            "C3": compute_patient_metrics(
                abnormal_targets, data.c3[selected], CLINICAL_CLASSES
            ),
            "M3": compute_patient_metrics(
                abnormal_targets, m3_conditional, CLINICAL_CLASSES
            ),
        },
        "interpretation": {
            "m2_above_m0": comparison_reports["M2_minus_M0"]["observed_delta"] > 0,
            "m2_above_m1": comparison_reports["M2_minus_M1"]["observed_delta"] > 0,
            "m3_above_m2": comparison_reports["M3_minus_M2"]["observed_delta"] > 0,
        },
    }
