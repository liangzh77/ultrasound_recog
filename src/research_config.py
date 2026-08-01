"""Validated YAML configuration contract for patient-level research runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_research_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Research config must be a mapping")
    if config.get("experiment_code") not in {"E0", "E1"}:
        raise ValueError("Only E0/E1 configs are supported by the mean baseline")
    if config.get("input_mode") not in {"full", "roi"}:
        raise ValueError("input_mode must be full or roi")
    expected_mode = {"E0": "full", "E1": "roi"}[config["experiment_code"]]
    if config["input_mode"] != expected_mode:
        raise ValueError("Experiment code and input_mode do not match")

    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    if data.get("output_size") != 384:
        raise ValueError("E0/E1 v1 must use 384 pixel inputs")
    if not 1 <= int(data.get("max_instances_train", 0)) <= 6:
        raise ValueError("max_instances_train must be between 1 and 6")
    if int(data.get("patient_batch_size", 0)) not in {1, 2}:
        raise ValueError("patient_batch_size must be 1 or 2")
    if int(data.get("num_workers", -1)) not in {0, 1, 2, 3}:
        raise ValueError("num_workers must be between 0 and 3")
    if model.get("num_classes") != 6:
        raise ValueError("The frozen primary diagnosis task has six classes")
    if int(training.get("max_epochs", 0)) > 60:
        raise ValueError("max_epochs cannot exceed 60")
    if int(training.get("pilot_epochs", 0)) > 5:
        raise ValueError("pilot_epochs cannot exceed 5")
    if float(runtime.get("target_hours", 100)) > 10:
        raise ValueError("target runtime cannot exceed 10 hours")
    if float(runtime.get("soft_limit_hours", 100)) > 11.5:
        raise ValueError("soft runtime limit cannot exceed 11.5 hours")
    if float(runtime.get("hard_limit_hours", 100)) > 23.5:
        raise ValueError("hard runtime limit cannot exceed 23.5 hours")
    if float(runtime.get("max_gpu_memory_gb", 100)) > 9.0:
        raise ValueError("GPU memory budget cannot exceed 9 GB")
    return config


def assert_configs_differ_only(
    first: dict[str, Any],
    second: dict[str, Any],
    allowed_top_level_keys: set[str],
) -> None:
    left = deepcopy(first)
    right = deepcopy(second)
    for key in allowed_top_level_keys:
        left.pop(key, None)
        right.pop(key, None)
    if left != right:
        raise ValueError("Comparable experiment configs change more than allowed fields")
