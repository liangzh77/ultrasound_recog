from pathlib import Path

import numpy as np
from PIL import Image

from src.research_proxy_audit import (
    GEOMETRY_FEATURES,
    PIXEL_SAMPLE_LIMIT,
    ProxyTable,
    _sample_strips,
    aggregate_patient_proxy_features,
    assess_proxy_risk,
    extract_image_proxy_features,
    permute_targets_within_folds,
    proxy_permutation_test,
    run_proxy_oof,
)


def test_extract_proxy_features_uses_only_registered_nonclinical_signals(tmp_path: Path):
    pixels = np.zeros((20, 30, 3), dtype=np.uint8)
    pixels[:, :5] = (255, 255, 255)
    pixels[4:16, 6:24] = (30, 80, 140)
    path = tmp_path / "a_diagnosis_like_name.jpg"
    Image.fromarray(pixels).save(path)

    features = extract_image_proxy_features(
        path,
        roi={"x1": 6, "y1": 4, "x2": 24, "y2": 16},
    )

    assert set(GEOMETRY_FEATURES) <= set(features)
    assert all(np.isfinite(value) for value in features.values())
    assert features["width"] == 30
    assert features["height"] == 20
    assert features["roi_area_fraction"] == 216 / 600
    assert features["is_jpeg"] == 1
    forbidden = {"diagnosis", "person", "path", "filename", "name"}
    assert not any(token in key.casefold() for key in features for token in forbidden)


def test_edge_statistics_never_materialize_more_than_fixed_sample_limit():
    strips = [np.zeros((500, 500, 3), dtype=np.uint8) for _ in range(4)]

    sampled = _sample_strips(strips)

    assert len(sampled) <= PIXEL_SAMPLE_LIMIT


def test_patient_aggregation_is_deterministic_and_rejects_mixed_metadata():
    rows = [
        {
            "person_key": "P2",
            "diagnosis_id": 1,
            "outer_fold": 1,
            "width": 10.0,
            "height": 20.0,
        },
        {
            "person_key": "P1",
            "diagnosis_id": 0,
            "outer_fold": 0,
            "width": 20.0,
            "height": 20.0,
        },
        {
            "person_key": "P1",
            "diagnosis_id": 0,
            "outer_fold": 0,
            "width": 40.0,
            "height": 30.0,
        },
    ]

    table = aggregate_patient_proxy_features(rows, ("width", "height"))

    assert table.person_keys == ("P1", "P2")
    assert table.targets.tolist() == [0, 1]
    assert table.outer_folds.tolist() == [0, 1]
    assert table.feature_names == (
        "image_count",
        "width_mean",
        "width_std",
        "width_min",
        "width_max",
        "height_mean",
        "height_std",
        "height_min",
        "height_max",
    )
    assert table.features[0].tolist() == [2, 30, 10, 20, 40, 25, 5, 20, 30]


def test_proxy_oof_covers_each_patient_once_with_valid_probabilities():
    rng = np.random.default_rng(17)
    targets = np.repeat(np.arange(3), 10)
    folds = np.tile(np.arange(5), 6)
    features = rng.normal(size=(30, 4))
    features[:, 0] += targets
    table = ProxyTable(
        person_keys=tuple(f"P{index:02d}" for index in range(30)),
        targets=targets,
        outer_folds=folds,
        features=features,
        feature_names=("a", "b", "c", "d"),
    )

    probabilities = run_proxy_oof(table, class_count=3, seed=23)

    assert probabilities.shape == (30, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_proxy_permutation_preserves_each_fold_class_distribution():
    targets = np.array([0, 0, 1, 1, 2, 2] * 2)
    folds = np.repeat([0, 1], 6)

    shuffled = permute_targets_within_folds(targets, folds, seed=31)

    for fold in (0, 1):
        mask = folds == fold
        assert sorted(shuffled[mask]) == sorted(targets[mask])


def test_proxy_risk_thresholds_are_fixed_before_real_audit():
    assert assess_proxy_risk(0.22, 0.60, 0.01) == "high"
    assert assess_proxy_risk(0.17, 0.61, 0.01) == "moderate"
    assert assess_proxy_risk(0.16, 0.55, 0.50) == "low"


def test_proxy_permutation_test_is_reproducible():
    rng = np.random.default_rng(41)
    targets = np.repeat(np.arange(3), 10)
    table = ProxyTable(
        person_keys=tuple(f"P{index:02d}" for index in range(30)),
        targets=targets,
        outer_folds=np.tile(np.arange(5), 6),
        features=rng.normal(size=(30, 3)),
        feature_names=("a", "b", "c"),
    )

    first = proxy_permutation_test(table, 3, observed_macro_f1=0.5, count=3, seed=7)
    second = proxy_permutation_test(table, 3, observed_macro_f1=0.5, count=3, seed=7)

    assert first == second
    assert 0 < first["p_value"] <= 1
    assert len(first["null_macro_f1"]) == 3
