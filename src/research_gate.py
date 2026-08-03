"""Frozen contracts for the G0 patient-level normal/abnormal image gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.research_dataset import ResearchImageRecord
from src.research_schema import DIAGNOSIS_CLASSES


GATE_CLASSES = ("normal", "abnormal")
GATE_CLASS_TO_ID = {name: index for index, name in enumerate(GATE_CLASSES)}
NORMAL_DIAGNOSIS = "正常"
ABNORMAL_DIAGNOSES = tuple(
    diagnosis for diagnosis in DIAGNOSIS_CLASSES if diagnosis != NORMAL_DIAGNOSIS
)
G0_DATA_FINGERPRINT = (
    "62ecb01c4d77ec0012704611ecc8d18ef51ebb4e0ea744fb3896948829f0b675"
)
G0_CONFIG_SHA256 = (
    "4a06e647e6b1ba5e4223a4ec752110bf71b9bfc257e8b134271c508c9c53ed72"
)


@dataclass(frozen=True)
class OperatingThreshold:
    threshold: float
    abnormal_sensitivity: float
    normal_specificity: float
    minimum_abnormal_sensitivity: float
    constraint_met: bool
    fit_split: str


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    validation_nll_before: float
    validation_nll_after: float
    optimizer_succeeded: bool
    used_identity_fallback: bool
    fit_split: str


@dataclass(frozen=True)
class GatePostprocessor:
    calibration: TemperatureCalibration
    operating_threshold: OperatingThreshold

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return apply_temperature(probabilities, self.calibration.temperature)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.transform(probabilities)
        return (
            calibrated[:, GATE_CLASS_TO_ID["abnormal"]]
            >= self.operating_threshold.threshold
        ).astype(np.int64)


def _nested(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing frozen G0 config field: {dotted_key}")
        value = value[part]
    return value


def _require_frozen_values(config: dict[str, Any]) -> None:
    expected = {
        "experiment_code": "G0",
        "status": "frozen_preregistered",
        "study_id": "knee_patient_multimodal_v1_20260724",
        "data_fingerprint": G0_DATA_FINGERPRINT,
        "input_mode": "roi",
        "seed": 20260724,
        "task.prediction_level": "patient",
        "task.type": "binary_normal_abnormal",
        "task.normal_source_class": NORMAL_DIAGNOSIS,
        "task.normal_id": 0,
        "task.abnormal_id": 1,
        "data.expected_patients": 967,
        "data.expected_images": 4543,
        "data.expected_normal_patients": 200,
        "data.expected_abnormal_patients": 767,
        "data.output_size": 384,
        "data.resize_mode": "letterbox",
        "data.max_instances_train": 6,
        "data.patient_batch_size": 1,
        "data.effective_patient_batch_size": 8,
        "data.num_workers": 2,
        "model.name": "efficientnet_b2.ra_in1k",
        "model.pretrained": True,
        "model.pretrained_path": "assets/pretrained/efficientnet_b2_ra-bcdf34b7.pth",
        "model.pretrained_sha256": (
            "bcdf34b7ab5a07e20e8cd37d74f7f40ca398b5105e06c755342a9c7ffa892944"
        ),
        "model.num_classes": 2,
        "model.aggregation": "gated_attention",
        "model.attention_dim": 256,
        "model.attention_collapse_threshold": 0.95,
        "model.max_multi_image_collapse_rate": 0.50,
        "model.dropout": 0.30,
        "optimizer.name": "AdamW",
        "optimizer.encoder_lr": 0.0001,
        "optimizer.head_lr": 0.0003,
        "optimizer.weight_decay": 0.0001,
        "optimizer.gradient_clip": 1.0,
        "training.max_epochs": 60,
        "training.warmup_epochs": 3,
        "training.early_stopping_patience": 10,
        "training.early_stopping_metric": "macro_f1",
        "training.amp": True,
        "training.pilot_epochs": 5,
        "training.attention_kl_weight": 0.05,
        "evaluation.bootstrap_samples": 2000,
        "evaluation.bootstrap_seed": 20260724,
        "evaluation.threshold_selection.fit_split": "inner_validation",
        "evaluation.threshold_selection.objective": "maximize_normal_specificity",
        "evaluation.threshold_selection.minimum_abnormal_sensitivity": 0.90,
        "evaluation.threshold_selection.fallback_threshold": 0.50,
        "evaluation.threshold_selection.fallback_is_gate_failure": True,
        "evaluation.calibration.method": "temperature_scaling",
        "evaluation.calibration.fit_split": "inner_validation",
        "evaluation.calibration.report_uncalibrated": True,
        "evaluation.calibration.formal_probabilities": "calibrated",
        "gates.minimum_oof_roc_auc": 0.80,
        "gates.minimum_oof_roc_auc_ci_low": 0.75,
        "gates.minimum_abnormal_sensitivity": 0.90,
        "gates.minimum_normal_specificity": 0.50,
        "gates.minimum_macro_f1": 0.70,
        "gates.minimum_folds_with_roc_auc_at_least_0_75": 4,
        "gates.maximum_calibrated_ece": 0.10,
        "gates.maximum_calibrated_brier": 0.15,
        "gates.calibration_must_not_worsen": True,
        "gates.maximum_multi_image_attention_collapse_rate": 0.50,
        "runtime.target_hours": 10.0,
        "runtime.soft_limit_hours": 11.5,
        "runtime.hard_limit_hours": 23.5,
        "runtime.max_gpu_memory_gb": 9.0,
        "runtime.max_cpu_threads": 4,
        "runtime.max_interop_threads": 1,
        "privacy.allow_clinical_features": False,
        "privacy.allow_excel_availability": False,
        "privacy.allow_raw_paths_in_public_outputs": False,
        "privacy.allow_source_filenames_in_public_outputs": False,
        "privacy.allow_2026_labels": False,
        "stopping.on_any_gate_failure": "stop_without_second_candidate",
        "stopping.prohibit_backbone_search": True,
        "stopping.prohibit_aggregation_search": True,
        "stopping.prohibit_loss_search": True,
        "stopping.prohibit_kl_weight_search": True,
        "stopping.prohibit_input_geometry_search": True,
    }
    for dotted_key, expected_value in expected.items():
        if _nested(config, dotted_key) != expected_value:
            raise ValueError(f"Frozen G0 config mismatch: {dotted_key}")


def load_gate_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("G0 config must be a mapping")
    _require_frozen_values(config)
    if tuple(_nested(config, "task.classes")) != GATE_CLASSES:
        raise ValueError("Frozen G0 class order changed")
    if tuple(_nested(config, "task.abnormal_source_classes")) != ABNORMAL_DIAGNOSES:
        raise ValueError("Frozen G0 abnormal diagnosis set changed")
    if tuple(_nested(config, "evaluation.outer_folds")) != tuple(range(5)):
        raise ValueError("Frozen G0 outer folds changed")
    if tuple(_nested(config, "evaluation.seeds")) != tuple(
        range(20260724, 20260729)
    ):
        raise ValueError("Frozen G0 seeds changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != G0_CONFIG_SHA256:
        raise ValueError("Frozen G0 config SHA-256 mismatch")
    return config


def diagnosis_to_gate_id(diagnosis: str) -> int:
    if diagnosis == NORMAL_DIAGNOSIS:
        return GATE_CLASS_TO_ID["normal"]
    if diagnosis in ABNORMAL_DIAGNOSES:
        return GATE_CLASS_TO_ID["abnormal"]
    raise ValueError("Diagnosis is outside the frozen G0 task")


def remap_records_to_gate(
    records: list[ResearchImageRecord],
) -> list[ResearchImageRecord]:
    remapped = []
    for record in records:
        gate_id = diagnosis_to_gate_id(record.diagnosis)
        source_id = DIAGNOSIS_CLASSES.index(record.diagnosis)
        if record.diagnosis_id != source_id:
            raise ValueError("Source diagnosis and diagnosis_id do not match")
        remapped.append(
            replace(
                record,
                diagnosis=GATE_CLASSES[gate_id],
                diagnosis_id=gate_id,
            )
        )
    return remapped


def _validate_binary_inputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probabilities.shape != (len(targets), 2):
        raise ValueError("G0 inputs must be one target and two probabilities per patient")
    if len(targets) == 0 or set(np.unique(targets)) != {0, 1}:
        raise ValueError("G0 evaluation requires both normal and abnormal patients")
    if not np.isfinite(probabilities).all():
        raise ValueError("G0 probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("G0 probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("G0 probabilities must sum to 1")
    return targets, probabilities


def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("G0 probabilities must have shape [patients, 2]")
    if len(probabilities) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("G0 probabilities must be non-empty and finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("G0 probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("G0 probabilities must sum to 1")
    return probabilities


def _sensitivity_specificity(
    targets: np.ndarray,
    abnormal_probabilities: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    predictions = (abnormal_probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return float(sensitivity), float(specificity)


def select_operating_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_abnormal_sensitivity: float,
    fit_split: str,
) -> OperatingThreshold:
    if fit_split != "inner_validation":
        raise ValueError("G0 threshold may only be fitted on inner_validation")
    if not 0 < minimum_abnormal_sensitivity <= 1:
        raise ValueError("minimum_abnormal_sensitivity must be in (0, 1]")
    targets, probabilities = _validate_binary_inputs(targets, probabilities)
    abnormal_probabilities = probabilities[:, 1]
    candidates = sorted(
        {0.0, 0.5, 1.0, *(float(value) for value in abnormal_probabilities)}
    )
    eligible = []
    for threshold in candidates:
        sensitivity, specificity = _sensitivity_specificity(
            targets, abnormal_probabilities, threshold
        )
        if sensitivity >= minimum_abnormal_sensitivity:
            eligible.append((specificity, sensitivity, threshold))
    if not eligible:
        return OperatingThreshold(
            threshold=0.5,
            abnormal_sensitivity=0.0,
            normal_specificity=0.0,
            minimum_abnormal_sensitivity=minimum_abnormal_sensitivity,
            constraint_met=False,
            fit_split=fit_split,
        )
    specificity, sensitivity, threshold = max(eligible)
    return OperatingThreshold(
        threshold=float(threshold),
        abnormal_sensitivity=float(sensitivity),
        normal_specificity=float(specificity),
        minimum_abnormal_sensitivity=minimum_abnormal_sensitivity,
        constraint_met=True,
        fit_split=fit_split,
    )


def _binary_ece(
    targets: np.ndarray,
    abnormal_probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (abnormal_probabilities >= edges[index]) & (
            abnormal_probabilities <= edges[index + 1]
            if upper_inclusive
            else abnormal_probabilities < edges[index + 1]
        )
        if selected.any():
            ece += selected.mean() * abs(
                float(targets[selected].mean())
                - float(abnormal_probabilities[selected].mean())
            )
    return float(ece)


def compute_gate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    targets, probabilities = _validate_binary_inputs(targets, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("G0 operating threshold must be in [0, 1]")
    abnormal_probabilities = probabilities[:, 1]
    predictions = (abnormal_probabilities >= threshold).astype(np.int64)
    sensitivity, specificity = _sensitivity_specificity(
        targets, abnormal_probabilities, threshold
    )
    return {
        "patients": int(len(targets)),
        "normal_patients": int((targets == 0).sum()),
        "abnormal_patients": int((targets == 1).sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(targets, abnormal_probabilities)),
        "pr_auc": float(average_precision_score(targets, abnormal_probabilities)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "abnormal_sensitivity": sensitivity,
        "normal_specificity": specificity,
        "binary_brier": float(
            np.mean(np.square(abnormal_probabilities - targets))
        ),
        "binary_ece": _binary_ece(
            targets, abnormal_probabilities, bins=calibration_bins
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=[0, 1]
        ).tolist(),
    }


def bootstrap_roc_auc_ci(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    targets, probabilities = _validate_binary_inputs(targets, probabilities)
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = np.random.default_rng(seed)
    indices_by_class = [np.flatnonzero(targets == class_id) for class_id in (0, 1)]
    estimates = []
    for _ in range(samples):
        sampled = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in indices_by_class
            ]
        )
        estimates.append(roc_auc_score(targets[sampled], probabilities[sampled, 1]))
    low, high = np.percentile(estimates, (2.5, 97.5))
    observed = roc_auc_score(targets, probabilities[:, 1])
    return float(low), float(observed), float(high)


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    probabilities = _validate_probability_matrix(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("G0 temperature must be finite and positive")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    scaled_logits = np.log(clipped) / float(temperature)
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled_logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _binary_nll(targets: np.ndarray, probabilities: np.ndarray) -> float:
    selected = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    fit_split: str,
) -> TemperatureCalibration:
    if fit_split != "inner_validation":
        raise ValueError("G0 temperature may only be fitted on inner_validation")
    targets, probabilities = _validate_binary_inputs(targets, probabilities)
    before = _binary_nll(targets, probabilities)

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(probabilities, float(np.exp(log_temperature)))
        return _binary_nll(targets, calibrated)

    optimized = minimize_scalar(
        objective,
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    candidate_temperature = float(np.exp(optimized.x))
    candidate_after = float(optimized.fun)
    use_identity = bool(not optimized.success or candidate_after > before + 1e-12)
    temperature = 1.0 if use_identity else candidate_temperature
    after = before if use_identity else candidate_after
    return TemperatureCalibration(
        temperature=temperature,
        validation_nll_before=before,
        validation_nll_after=after,
        optimizer_succeeded=bool(optimized.success),
        used_identity_fallback=use_identity,
        fit_split=fit_split,
    )


def fit_gate_postprocessor(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    minimum_abnormal_sensitivity: float,
) -> GatePostprocessor:
    calibration = fit_temperature(
        validation_targets,
        validation_probabilities,
        fit_split="inner_validation",
    )
    calibrated = apply_temperature(
        validation_probabilities,
        calibration.temperature,
    )
    threshold = select_operating_threshold(
        validation_targets,
        calibrated,
        minimum_abnormal_sensitivity=minimum_abnormal_sensitivity,
        fit_split="inner_validation",
    )
    return GatePostprocessor(calibration=calibration, operating_threshold=threshold)


def build_gate_prediction_rows(
    *,
    person_keys: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    image_counts: Sequence[int],
    outer_fold: int,
    model_id: str,
    postprocessor: GatePostprocessor,
) -> list[dict[str, Any]]:
    targets, probabilities = _validate_binary_inputs(targets, probabilities)
    if len(person_keys) != len(targets) or len(image_counts) != len(targets):
        raise ValueError("G0 prediction metadata must align one-to-one with patients")
    if len(person_keys) != len(set(person_keys)):
        raise ValueError("G0 prediction person_key values must be unique")
    if any(int(value) < 1 for value in image_counts):
        raise ValueError("Every G0 prediction requires at least one image")
    if outer_fold not in range(5):
        raise ValueError("G0 outer_fold must be between 0 and 4")
    if not model_id.strip():
        raise ValueError("G0 model_id cannot be empty")
    calibrated = postprocessor.transform(probabilities)
    predictions = (
        calibrated[:, GATE_CLASS_TO_ID["abnormal"]]
        >= postprocessor.operating_threshold.threshold
    ).astype(np.int64)
    rows = []
    for index, person_key in enumerate(person_keys):
        rows.append(
            {
                "prediction_level": "patient_gate",
                "person_key": person_key,
                "outer_fold": outer_fold,
                "reference_class": GATE_CLASSES[int(targets[index])],
                "reference_id": int(targets[index]),
                "raw_prob_normal": float(probabilities[index, 0]),
                "raw_prob_abnormal": float(probabilities[index, 1]),
                "prob_normal": float(calibrated[index, 0]),
                "prob_abnormal": float(calibrated[index, 1]),
                "operating_threshold": postprocessor.operating_threshold.threshold,
                "predicted_class": GATE_CLASSES[int(predictions[index])],
                "predicted_id": int(predictions[index]),
                "temperature": postprocessor.calibration.temperature,
                "image_count": int(image_counts[index]),
                "model_id": model_id,
            }
        )
    return sorted(rows, key=lambda row: row["person_key"])
