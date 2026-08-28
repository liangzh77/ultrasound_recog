import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.research_dataset import ResearchImageRecord
from src.research_semantic_contours import (
    SemanticContourResearchImageDataset,
    load_semantic_contour_manifest,
    render_category_masks,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_category_masks_preserve_overlap_and_channel_identity():
    annotation = {
        "info": {"width": 10, "height": 8},
        "objects": [
            {"category": "积液", "segmentation": [[2, 1], [7, 1], [7, 6], [2, 6]]},
            {"category": "滑膜血流", "segmentation": [[4, 3], [8, 3], [8, 7], [4, 7]]},
            {"category": "骨皮质", "segmentation": [[1, 1], [2, 1], [2, 2]]},
        ],
    }
    masks, counts, pixels = render_category_masks(
        annotation,
        {"x1": 1, "y1": 0, "x2": 9, "y2": 8},
        ("积液", "滑膜血流"),
        output_size=16,
    )

    first = np.asarray(masks["积液"])
    second = np.asarray(masks["滑膜血流"])
    assert counts == {"积液": 1, "滑膜血流": 1}
    assert pixels["积液"] > 0 and pixels["滑膜血流"] > 0
    assert np.logical_and(first > 0, second > 0).any()


def test_sparse_manifest_loads_ordered_dataset_channels(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(100, 120, 140)).save(image_path)
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (8, 8), color=0)
    mask.putpixel((2, 3), 255)
    mask.save(mask_path)
    manifest_path = tmp_path / "manifest.csv"
    payload = {
        "积液": {
            "path": "mask.png",
            "sha256": _sha256(mask_path),
            "object_count": 1,
            "positive_pixels": 1,
        }
    }
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_key", "category_masks_json"))
        writer.writeheader()
        writer.writerow(
            {"image_key": "I1", "category_masks_json": json.dumps(payload, ensure_ascii=False)}
        )
    records = load_semantic_contour_manifest(
        manifest_path,
        tmp_path,
        _sha256(manifest_path),
        {"I1"},
        ("积液", "骨侵蚀"),
    )
    image_record = ResearchImageRecord(
        image_key="I1",
        person_key="P1",
        diagnosis="正常",
        diagnosis_id=0,
        image_path=image_path,
        roi={"x1": 0, "y1": 0, "x2": 8, "y2": 8},
    )
    dataset = SemanticContourResearchImageDataset(
        [image_record], records, ("积液", "骨侵蚀"), output_size=8, normalize=False
    )

    tensor = dataset[0]["image"]
    assert tensor.shape == (5, 8, 8)
    assert tensor[3, 3, 2] == 1
    assert tensor[4].sum() == 0
    assert tensor[:3, 7, 7].sum() > 0


def test_sparse_manifest_rejects_unknown_category(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (4, 4), color=255).save(mask_path)
    manifest_path = tmp_path / "manifest.csv"
    payload = {
        "未知": {
            "path": "mask.png",
            "sha256": _sha256(mask_path),
            "object_count": 1,
            "positive_pixels": 16,
        }
    }
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_key", "category_masks_json"))
        writer.writeheader()
        writer.writerow({"image_key": "I1", "category_masks_json": json.dumps(payload, ensure_ascii=False)})

    with pytest.raises(ValueError, match="unknown"):
        load_semantic_contour_manifest(
            manifest_path, tmp_path, _sha256(manifest_path), {"I1"}, ("积液",)
        )
