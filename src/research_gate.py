"""Frozen contracts for the G0 patient-level normal/abnormal image gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

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
