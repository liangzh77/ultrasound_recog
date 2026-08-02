"""Patient-level metrics and paired uncertainty estimates for the study."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def validate_oof_predictions(
    person_keys: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    expected_person_keys: Collection[str],
    prediction_level: str,
) -> None:
    if prediction_level != "patient":
        raise ValueError("OOF predictions must be patient-level")
    if len(person_keys) != len(set(person_keys)):
        raise ValueError("OOF person_key values must be unique")
    if set(person_keys) != set(expected_person_keys):
        missing = set(expected_person_keys) - set(person_keys)
        extra = set(person_keys) - set(expected_person_keys)
        raise ValueError(f"OOF coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or len(targets) != len(person_keys):
        raise ValueError("targets must contain one value per patient")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(person_keys):
        raise ValueError("probabilities must be [patients, classes]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1")
    if (targets < 0).any() or (targets >= probabilities.shape[1]).any():
        raise ValueError("targets are outside the probability class range")


def _expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not selected.any():
            continue
        result += selected.mean() * abs(
            float(correct[selected].mean()) - float(confidence[selected].mean())
        )
    return float(result)


def compute_patient_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    calibration_bins: int = 10,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_count = len(class_names)
    if probabilities.shape != (len(targets), class_count):
        raise ValueError("probability shape does not match targets/classes")
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(class_count)
    matrix = confusion_matrix(targets, predictions, labels=labels)
    per_class = []
    for class_id, class_name in enumerate(class_names):
        true_positive = int(matrix[class_id, class_id])
        false_negative = int(matrix[class_id, :].sum() - true_positive)
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_negative - false_positive)
        sensitivity_denominator = true_positive + false_negative
        specificity_denominator = true_negative + false_positive
        precision_denominator = true_positive + false_positive
        npv_denominator = true_negative + false_negative
        binary_targets = (targets == class_id).astype(np.int64)
        auc = (
            float(roc_auc_score(binary_targets, probabilities[:, class_id]))
            if np.unique(binary_targets).size == 2
            else None
        )
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": int(sensitivity_denominator),
                "sensitivity": (
                    true_positive / sensitivity_denominator
                    if sensitivity_denominator
                    else None
                ),
                "specificity": (
                    true_negative / specificity_denominator
                    if specificity_denominator
                    else None
                ),
                "precision": (
                    true_positive / precision_denominator
                    if precision_denominator
                    else None
                ),
                "npv": true_negative / npv_denominator if npv_denominator else None,
                "f1": float(
                    f1_score(
                        binary_targets,
                        predictions == class_id,
                        zero_division=0,
                    )
                ),
                "auc": auc,
            }
        )
    auc_values = [row["auc"] for row in per_class if row["auc"] is not None]
    sensitivities = [
        row["sensitivity"] for row in per_class if row["sensitivity"] is not None
    ]
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    top_two = np.argsort(probabilities, axis=1)[:, -min(2, class_count) :]
    return {
        "patients": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(np.mean(sensitivities)),
        "top2_accuracy": float(
            np.mean([target in choices for target, choices in zip(targets, top_two)])
        ),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_auc": float(np.mean(auc_values)) if auc_values else None,
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(targets, probabilities, calibration_bins),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def _stratified_bootstrap_indices(
    targets: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    samples = []
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        samples.append(generator.choice(indices, size=len(indices), replace=True))
    return np.concatenate(samples)


def _macro_f1(targets: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.arange(probabilities.shape[1])
    return float(
        f1_score(
            targets,
            probabilities.argmax(axis=1),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_macro_f1_ci(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = [
        _macro_f1(targets[indices], probabilities[indices])
        for indices in (
            _stratified_bootstrap_indices(targets, generator)
            for _ in range(n_bootstrap)
        )
    ]
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), _macro_f1(targets, probabilities), float(high)


def _macro_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    auc_values = []
    for class_id in range(probabilities.shape[1]):
        binary_targets = (targets == class_id).astype(np.int64)
        if np.unique(binary_targets).size < 2:
            continue
        auc_values.append(roc_auc_score(binary_targets, probabilities[:, class_id]))
    if not auc_values:
        raise ValueError("Macro AUC requires at least one evaluable class")
    return float(np.mean(auc_values))


def bootstrap_macro_auc_ci(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return a stratified patient bootstrap interval for one-vs-rest macro AUC."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = [
        _macro_auc(targets[indices], probabilities[indices])
        for indices in (
            _stratified_bootstrap_indices(targets, generator)
            for _ in range(n_bootstrap)
        )
    ]
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), _macro_auc(targets, probabilities), float(high)


def paired_bootstrap_macro_f1(
    targets: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError("Paired model probabilities must have identical shape")
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        indices = _stratified_bootstrap_indices(targets, generator)
        deltas.append(
            _macro_f1(targets[indices], candidate[indices])
            - _macro_f1(targets[indices], baseline[indices])
        )
    low, high = np.percentile(deltas, (2.5, 97.5))
    return {
        "observed_delta": _macro_f1(targets, candidate) - _macro_f1(targets, baseline),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_delta_gt_zero": float(np.mean(np.asarray(deltas) > 0)),
    }
