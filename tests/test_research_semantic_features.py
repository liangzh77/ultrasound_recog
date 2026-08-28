from pathlib import Path

import numpy as np

from src.research_semantic_contours import CategoryMask, SemanticContourRecord
from src.research_semantic_features import (
    build_patient_semantic_features,
    fit_semantic_feature_oof,
)


def _contour(image_key, category=None, objects=0, pixels=0):
    masks = {}
    if category is not None:
        masks[category] = CategoryMask(
            path=Path("unused.png"),
            sha256="0" * 64,
            object_count=objects,
            positive_pixels=pixels,
        )
    return SemanticContourRecord(image_key=image_key, category_masks=masks)


def test_patient_features_aggregate_presence_count_and_total_canvas_area():
    rows = [
        {"image_key": "I1", "person_key": "P1", "diagnosis": "正常", "diagnosis_id": "0", "outer_fold": "0"},
        {"image_key": "I2", "person_key": "P1", "diagnosis": "正常", "diagnosis_id": "0", "outer_fold": "0"},
    ]
    contours = {
        "I1": _contour("I1", "积液", objects=2, pixels=10),
        "I2": _contour("I2", "积液", objects=1, pixels=30),
    }
    table = build_patient_semantic_features(
        rows, contours, ("积液", "骨侵蚀"), canvas_pixels=100
    )

    assert table.person_keys == ("P1",)
    assert table.feature_names == (
        "积液__presence",
        "积液__object_count",
        "积液__area_fraction",
        "骨侵蚀__presence",
        "骨侵蚀__object_count",
        "骨侵蚀__area_fraction",
    )
    assert np.allclose(table.features[0], [1, 3, 0.2, 0, 0, 0])


def test_semantic_feature_oof_is_complete_with_six_classes_and_five_folds():
    rows = []
    contours = {}
    for fold in range(5):
        for diagnosis_id in range(6):
            for repeat in range(2):
                key = f"P{fold}-{diagnosis_id}-{repeat}"
                image_key = f"I{fold}-{diagnosis_id}-{repeat}"
                rows.append(
                    {
                        "image_key": image_key,
                        "person_key": key,
                        "diagnosis": str(diagnosis_id),
                        "diagnosis_id": str(diagnosis_id),
                        "outer_fold": str(fold),
                    }
                )
                contours[image_key] = _contour(
                    image_key,
                    "积液" if diagnosis_id % 2 else None,
                    objects=1,
                    pixels=diagnosis_id + 1,
                )
    table = build_patient_semantic_features(
        rows, contours, ("积液",), canvas_pixels=100
    )
    probabilities, models = fit_semantic_feature_oof(table, seed=1)

    assert probabilities.shape == (60, 6)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert len(models) == 5
