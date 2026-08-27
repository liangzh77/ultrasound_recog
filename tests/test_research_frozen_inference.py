import numpy as np
import pytest

from src.research_auto_roi import ROIBox
from src.research_frozen_inference import (
    audit_numbered_image_cohort,
    confidence_flag,
    hierarchical_probabilities,
    mean_probability_vectors,
    median_roi_box,
)


def test_mean_probability_vectors_is_equal_weighted_and_validated():
    result = mean_probability_vectors(
        [[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]], expected_columns=2
    )
    assert result == pytest.approx((0.6, 0.4))
    with pytest.raises(ValueError, match="sum to one"):
        mean_probability_vectors([[0.8, 0.3]], expected_columns=2)


def test_median_roi_box_uses_available_boxes_and_clamps():
    result = median_roi_box(
        [ROIBox(-2, 5, 90, 70), ROIBox(10, 7, 110, 75), ROIBox(8, 6, 92, 74)],
        width=100,
        height=80,
    )
    assert result == ROIBox(8.0, 6.0, 92.0, 74.0)
    assert median_roi_box([], width=100, height=80) is None


def test_hierarchical_probabilities_preserve_image_normal_mass():
    combined = hierarchical_probabilities(
        [0.4, 0.1, 0.1, 0.1, 0.2, 0.1],
        [0.5, 0.2, 0.1, 0.1, 0.1],
    )
    assert combined == pytest.approx((0.4, 0.3, 0.12, 0.06, 0.06, 0.06))
    assert np.sum(combined) == pytest.approx(1.0)


def test_confidence_flag_reports_both_frozen_reasons():
    flagged, reasons = confidence_flag([0.45, 0.40, 0.10, 0.05])
    assert flagged is True
    assert reasons == (
        "top1_probability_below_threshold",
        "top1_top2_margin_below_threshold",
    )
    assert confidence_flag([0.7, 0.2, 0.1]) == (False, ())


def test_audit_numbered_image_cohort_checks_exact_coverage(tmp_path):
    for patient in ("1", "2"):
        patient_dir = tmp_path / patient
        patient_dir.mkdir()
        (patient_dir / f"{patient}.jpg").write_bytes(b"jpg")
    assert audit_numbered_image_cohort(
        tmp_path, expected_patients=2, expected_images=2
    ) == {"patients": 2, "images": 2, "all_patients_have_images": True}
    (tmp_path / "2" / "2.jpg").unlink()
    with pytest.raises(ValueError, match="expected 2 blind images"):
        audit_numbered_image_cohort(tmp_path, expected_patients=2, expected_images=2)

