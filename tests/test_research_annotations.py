from copy import deepcopy

from src.research_annotations import (
    annotation_geometry_signature,
    normalize_annotation,
)


def test_normalization_changes_only_region_names_not_geometry():
    source = {
        "info": {"name": "example.jpg", "width": 100, "height": 80},
        "objects": [
            {
                "category": "OA骨皮质",
                "segmentation": [[1.0, 2.0], [5.0, 2.0], [5.0, 6.0]],
                "bbox": [1.0, 2.0, 5.0, 6.0],
                "area": 8.0,
            },
            {
                "category": "OA骨赘",
                "segmentation": [[10.0, 20.0], [15.0, 20.0], [15.0, 26.0]],
            },
        ],
        "ultrasound_rect": {"x1": 2, "y1": 3, "x2": 90, "y2": 70},
        "ultrasound_rect_reviewed": True,
    }
    original = deepcopy(source)

    normalized, changes = normalize_annotation(source, "骨性关节炎")

    assert source == original
    assert [item["category"] for item in normalized["objects"]] == [
        "骨皮质",
        "骨赘",
    ]
    assert changes == {"OA骨皮质": "骨皮质", "OA骨赘": "骨赘"}
    assert annotation_geometry_signature(normalized) == (
        annotation_geometry_signature(source)
    )
