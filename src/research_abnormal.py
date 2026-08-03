"""Frozen abnormal-only five-class contract for the D0 image experiment."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research_clinical import CLINICAL_CLASSES
from src.research_dataset import ResearchImageRecord
from src.research_schema import DIAGNOSIS_CLASSES


D0_CONFIG_SHA256 = "6c7da2c0e83924bcb20ac75c4c54cdf32a5b57dee017ab21ce9de7b4f4683e0d"
D0_DATA_FINGERPRINT = "62ecb01c4d77ec0012704611ecc8d18ef51ebb4e0ea744fb3896948829f0b675"
D0_CLASS_SLUGS = ("ra", "ga", "spa", "oa", "injury")
D0_PROBABILITY_COLUMNS = tuple(f"prob_{slug}" for slug in D0_CLASS_SLUGS)


def _validate_d0_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_code") != "D0" or config.get("status") != "frozen_preregistered":
        raise ValueError("D0 config identity or status changed")
    if config.get("data_fingerprint") != D0_DATA_FINGERPRINT:
        raise ValueError("D0 data fingerprint changed")
    if config.get("input_mode") != "roi":
        raise ValueError("D0 must use ROI input")
    task = config.get("task", {})
    if task.get("type") != "abnormal_five_class":
        raise ValueError("D0 task type changed")
    if tuple(task.get("classes", ())) != D0_CLASS_SLUGS:
        raise ValueError("D0 class order changed")
    if tuple(task.get("source_classes", ())) != CLINICAL_CLASSES:
        raise ValueError("D0 source diagnoses changed")
    if task.get("excluded_source_class") != DIAGNOSIS_CLASSES[0]:
        raise ValueError("D0 excluded normal class changed")

    data = config.get("data", {})
    if int(data.get("expected_patients", 0)) != 767 or int(data.get("expected_images", 0)) != 3789:
        raise ValueError("D0 expected cohort size changed")
    if sum(int(value) for value in data.get("expected_patient_counts", {}).values()) != 767:
        raise ValueError("D0 expected patient counts do not sum to 767")
    if sum(int(value) for value in data.get("expected_image_counts", {}).values()) != 3789:
        raise ValueError("D0 expected image counts do not sum to 3789")
    if data.get("output_size") != 384 or data.get("resize_mode") != "letterbox":
        raise ValueError("D0 input geometry changed")

    model = config.get("model", {})
    if (
        model.get("name") != "efficientnet_b2.ra_in1k"
        or int(model.get("num_classes", 0)) != 5
        or model.get("aggregation") != "gated_attention"
        or int(model.get("attention_dim", 0)) != 256
    ):
        raise ValueError("D0 frozen model contract changed")
    if float(config.get("training", {}).get("attention_kl_weight", -1)) != 0.05:
        raise ValueError("D0 attention KL weight changed")
    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("outer_folds", ())) != tuple(range(5)):
        raise ValueError("D0 outer folds changed")
    if tuple(evaluation.get("seeds", ())) != tuple(range(20260724, 20260729)):
        raise ValueError("D0 seeds changed")
    if int(config.get("runtime", {}).get("max_cpu_threads", 0)) != 2:
        raise ValueError("D0 CPU thread limit changed")


def load_d0_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("D0 config must be a mapping")
    _validate_d0_config(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != D0_CONFIG_SHA256:
        raise ValueError("Frozen D0 config SHA-256 mismatch")
    return payload


def remap_records_to_abnormal(
    records: list[ResearchImageRecord],
) -> list[ResearchImageRecord]:
    remapped = []
    for record in records:
        if record.diagnosis == DIAGNOSIS_CLASSES[0]:
            continue
        if record.diagnosis not in CLINICAL_CLASSES:
            raise ValueError("D0 record diagnosis is outside the frozen task")
        source_id = DIAGNOSIS_CLASSES.index(record.diagnosis)
        if record.diagnosis_id != source_id:
            raise ValueError("D0 source diagnosis and ID differ")
        target_id = CLINICAL_CLASSES.index(record.diagnosis)
        remapped.append(replace(record, diagnosis_id=target_id))
    return remapped


def validate_d0_record_sets(
    record_sets: Mapping[str, list[ResearchImageRecord]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    records = [record for split in record_sets.values() for record in split]
    people: dict[str, tuple[str, int]] = {}
    for record in records:
        if record.diagnosis not in CLINICAL_CLASSES or record.diagnosis_id not in range(5):
            raise ValueError("D0 remapped record is outside the five-class contract")
        previous = people.setdefault(
            record.person_key, (record.diagnosis, record.diagnosis_id)
        )
        if previous != (record.diagnosis, record.diagnosis_id):
            raise ValueError("D0 patient has mixed diagnoses")
    if len({record.image_key for record in records}) != len(records):
        raise ValueError("D0 record sets contain duplicate images")

    patient_counts = Counter(diagnosis for diagnosis, _ in people.values())
    image_counts = Counter(record.diagnosis for record in records)
    expected = config["data"]
    if len(people) != int(expected["expected_patients"]) or len(records) != int(expected["expected_images"]):
        raise ValueError("D0 patient or image count differs from the frozen contract")
    if dict(patient_counts) != {
        str(key): int(value) for key, value in expected["expected_patient_counts"].items()
    }:
        raise ValueError("D0 per-class patient counts differ from the frozen contract")
    if dict(image_counts) != {
        str(key): int(value) for key, value in expected["expected_image_counts"].items()
    }:
        raise ValueError("D0 per-class image counts differ from the frozen contract")
    return {
        "patients": len(people),
        "images": len(records),
        "patient_counts": dict(patient_counts),
        "image_counts": dict(image_counts),
    }
