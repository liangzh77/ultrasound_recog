from src.research_annotation_audit import audit_annotation_records


def _annotation(objects):
    return {
        "info": {"width": 100, "height": 80},
        "ultrasound_rect": {"x1": 5, "y1": 5, "x2": 95, "y2": 75},
        "objects": objects,
    }


def _object(category, points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "category": category,
        "segmentation": points,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "area": 100.0,
    }


def test_annotation_audit_counts_coverage_support_and_cross_class_overlap():
    image_rows = [
        {
            "image_key": "i1",
            "person_key": "p1",
            "diagnosis": "正常",
            "diagnosis_id": 0,
            "outer_fold": 0,
            "include": 1,
            "width": 100,
            "height": 80,
        },
        {
            "image_key": "i2",
            "person_key": "p2",
            "diagnosis": "损伤",
            "diagnosis_id": 5,
            "outer_fold": 1,
            "include": 1,
            "width": 100,
            "height": 80,
        },
    ]
    annotations = {
        "i1": _annotation(
            [
                _object("骨皮质", [[10, 10], [30, 10], [30, 30], [10, 30]]),
                _object("积液", [[20, 20], [40, 20], [40, 40], [20, 40]]),
            ]
        )
    }

    report, patient_labels = audit_annotation_records(
        image_rows,
        annotations,
        ["骨皮质", "积液"],
        {"骨皮质": "anatomy", "积液": "finding"},
        {
            "robust_min_patients": 1,
            "robust_min_patients_per_fold": 0,
            "limited_min_patients": 1,
            "limited_min_patients_per_fold": 0,
        },
    )

    assert report["coverage"]["images_with_annotation_json"] == 1
    assert report["coverage"]["images_without_annotation_json"] == 1
    assert report["polygon_overlaps"]["cross_category_pairs"] == 1
    assert report["polygon_overlaps"]["single_multiclass_mask_safe"] is False
    assert patient_labels["p1"] == {"骨皮质", "积液"}
    assert patient_labels.get("p2", set()) == set()
    assert report["categories"][0]["support_tier"] == "robust_multifold"


def test_annotation_audit_rejects_incomplete_role_contract():
    try:
        audit_annotation_records([], {}, ["骨皮质"], {})
    except ValueError as error:
        assert "configured role" in str(error)
    else:
        raise AssertionError("Missing category roles must be rejected")
