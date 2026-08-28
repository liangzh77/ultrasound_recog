import numpy as np

from src.research_semantic_study import (
    MSeriesData,
    evaluate_m_series,
    fuse_image_and_clinical,
    hierarchical_image_c3_comparator,
)


def test_fixed_fusion_keeps_normal_gate_and_blends_abnormal_conditionals():
    image = np.asarray(
        [
            [0.2, 0.4, 0.1, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
        ]
    )
    clinical = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    fused = fuse_image_and_clinical(
        image, clinical, np.asarray([True, False]), 0.75, 0.25
    )

    assert np.isclose(fused[0, 0], 0.2)
    assert np.allclose(fused[0, 1:], 0.8 * (0.75 * clinical[0] + 0.25 * image[0, 1:] / 0.8))
    assert np.allclose(fused[1], image[1])
    assert np.allclose(fused.sum(axis=1), 1)


def test_hierarchical_comparator_uses_only_c3_abnormal_conditional():
    image = np.asarray([[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]])
    clinical = np.asarray([[0.1, 0.2, 0.3, 0.25, 0.15]])
    result = hierarchical_image_c3_comparator(image, clinical, np.asarray([True]))

    assert np.isclose(result[0, 0], 0.25)
    assert np.allclose(result[0, 1:], 0.75 * clinical[0])


def test_evaluation_does_not_mutate_frozen_oof_probabilities():
    probabilities = np.asarray(
        [
            [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.7, 0.1, 0.05, 0.05, 0.05],
            [0.05, 0.1, 0.7, 0.05, 0.05, 0.05],
            [0.05, 0.1, 0.05, 0.7, 0.05, 0.05],
            [0.05, 0.1, 0.05, 0.05, 0.7, 0.05],
            [0.05, 0.1, 0.05, 0.05, 0.05, 0.7],
        ],
        dtype=np.float64,
    )
    clinical = np.full((6, 5), np.nan)
    clinical[1:] = np.eye(5)
    available = np.asarray([False, True, True, True, True, True])
    data = MSeriesData(
        person_keys=tuple(f"p{index}" for index in range(6)),
        references=("normal", "ra", "ga", "spa", "oa", "injury"),
        targets=np.arange(6),
        outer_folds=np.zeros(6, dtype=np.int64),
        image_counts=np.ones(6, dtype=np.int64),
        clinical_available=available,
        m0=probabilities.copy(),
        m1=probabilities.copy(),
        m2=probabilities.copy(),
        c3=clinical,
        m0_c3=probabilities.copy(),
        m3=probabilities.copy(),
    )
    before_m2 = data.m2.copy()
    before_m3 = data.m3.copy()

    evaluate_m_series(data, bootstrap_samples=10, bootstrap_seed=7)

    assert np.array_equal(data.m2, before_m2)
    assert np.array_equal(data.m3, before_m3)
