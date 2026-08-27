import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "tool_62_l0_mask_only_proxy",
    ROOT / "tools/62_audit_l0_mask_only_proxy.py",
)
proxy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(proxy)


def test_patient_proxy_features_preserve_only_presence_and_area_summaries():
    images = [
        {"include": "1", "image_key": "I1", "person_key": "P1"},
        {"include": "1", "image_key": "I2", "person_key": "P1"},
        {"include": "1", "image_key": "I3", "person_key": "P2"},
        {"include": "0", "image_key": "IX", "person_key": "PX"},
    ]
    masks = [
        {"image_key": "I1", "positive_pixels": "0"},
        {"image_key": "I2", "positive_pixels": "25"},
        {"image_key": "I3", "positive_pixels": "0"},
    ]

    result = proxy.build_patient_features(images, masks, mask_pixels=100)

    assert result["P1"] == {
        "positive_mask_image_fraction": 0.5,
        "mean_mask_area_fraction_all_images": 0.125,
        "maximum_mask_area_fraction": 0.25,
        "image_count": 2,
        "all_zero_mask": 0,
    }
    assert result["P2"]["all_zero_mask"] == 1


def test_proxy_cross_validation_predicts_each_outer_fold_once():
    features = np.asarray([[index % 3, index / 10, index / 20] for index in range(60)])
    targets = np.asarray([index % 6 for index in range(60)])
    folds = np.asarray([index % 5 for index in range(60)])
    config = {
        "evaluation": {"outer_folds": [0, 1, 2, 3, 4]},
        "model": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 20260827,
        },
    }

    probabilities, models = proxy.fit_proxy_oof(features, targets, folds, config)

    assert probabilities.shape == (60, 6)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert [item["outer_fold"] for item in models] == [0, 1, 2, 3, 4]
