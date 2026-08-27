"""Pure aggregation contracts for the frozen R2/2026 inference route."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.research_auto_roi import ROIBox


def _probability_matrix(
    vectors: Iterable[Sequence[float]], expected_columns: int
) -> np.ndarray:
    matrix = np.asarray(list(vectors), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"expected one or more {expected_columns}-class probability vectors"
        )
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability vector must sum to one")
    return matrix


def mean_probability_vectors(
    vectors: Iterable[Sequence[float]], expected_columns: int
) -> tuple[float, ...]:
    """Equally average frozen model probability vectors."""

    mean = _probability_matrix(vectors, expected_columns).mean(axis=0)
    return tuple(float(value) for value in mean)


def median_roi_box(
    boxes: Iterable[ROIBox], width: int, height: int
) -> ROIBox | None:
    """Coordinate-wise median of available detector boxes, clamped to the image."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    box_list = list(boxes)
    if not box_list:
        return None
    coordinates = np.asarray(
        [[box.x1, box.y1, box.x2, box.y2] for box in box_list], dtype=np.float64
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("ROI coordinates must be finite")
    x1, y1, x2, y2 = np.median(coordinates, axis=0)
    result = ROIBox(
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    )
    result.validate(width, height)
    return result


def hierarchical_probabilities(
    image_six_class: Sequence[float], clinical_abnormal_five_class: Sequence[float]
) -> tuple[float, ...]:
    """Combine image normal probability with conditional C3 abnormal probabilities."""

    image = _probability_matrix([image_six_class], 6)[0]
    clinical = _probability_matrix([clinical_abnormal_five_class], 5)[0]
    normal_probability = float(image[0])
    combined = np.concatenate(
        ([normal_probability], (1.0 - normal_probability) * clinical)
    )
    return tuple(float(value) for value in combined)


def confidence_flag(
    probabilities: Sequence[float],
    *,
    top1_threshold: float = 0.50,
    margin_threshold: float = 0.10,
) -> tuple[bool, tuple[str, ...]]:
    """Flag uncertain predictions without altering the predicted class."""

    matrix = _probability_matrix([probabilities], len(probabilities))[0]
    if matrix.size < 2:
        raise ValueError("confidence flag requires at least two classes")
    ordered = np.sort(matrix)[::-1]
    reasons = []
    if ordered[0] < top1_threshold:
        reasons.append("top1_probability_below_threshold")
    if ordered[0] - ordered[1] < margin_threshold:
        reasons.append("top1_top2_margin_below_threshold")
    return bool(reasons), tuple(reasons)

