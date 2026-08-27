import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "tool_64_2026_frozen_prediction",
    ROOT / "tools/64_run_2026_frozen_prediction.py",
)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


def test_final_rows_keeps_image_and_hierarchical_predictions_separate():
    person_keys = ("P1",)
    image_counts = np.asarray([2])
    image_probabilities = np.asarray([[0.3, 0.25, 0.15, 0.1, 0.1, 0.1]])
    c3_rows = {
        "P1": {
            "prob_ra": "0.1",
            "prob_ga": "0.6",
            "prob_spa": "0.1",
            "prob_oa": "0.1",
            "prob_injury": "0.1",
        }
    }
    roi_rows = [
        {"person_key": "P1", "full_image_fallback": 0, "available_r0_members": 5},
        {"person_key": "P1", "full_image_fallback": 1, "available_r0_members": 0},
    ]

    row = tool.final_rows(
        person_keys, image_counts, image_probabilities, c3_rows, roi_rows
    )[0]

    assert row["image_top1"] == "正常"
    assert row["hierarchical_top1"] == "痛风性关节炎"
    assert row["hierarchical_prob_normal"] == 0.3
    assert np.isclose(row["hierarchical_prob_ga"], 0.42)
    assert row["roi_full_image_fallback_images"] == 1
    assert row["roi_min_available_r0_members"] == 0


def test_definition_accepts_yaml_integer_or_string_fold_keys():
    assert tool.definition({0: {"value": 1}}, 0) == {"value": 1}
    assert tool.definition({"0": {"value": 2}}, 0) == {"value": 2}


def test_r2_freezes_small_r0_inference_batches_after_resource_failure():
    config = yaml.safe_load(tool.CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["auto_roi_ensemble"]["inference_batch_size"] == 8
    assert config["controls"]["resource_retry_may_only_change_inference_batching"] is True
