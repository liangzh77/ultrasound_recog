import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.research_dataset import ResearchImageRecord
from src.research_lesion_guided import (
    LesionGuidedResearchImageDataset,
    LesionMaskRecord,
    load_lesion_mask_manifest,
    render_binary_finding_mask,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_render_mask_keeps_all_findings_in_the_shared_effective_view_frame():
    annotation = {
        "info": {"width": 10, "height": 8},
        "objects": [
            {
                "category": "积液",
                "segmentation": [[2, 1], [5, 1], [5, 4], [2, 4]],
            },
            {
                "category": "积液",
                "segmentation": [[6, 5], [8, 5], [8, 7], [6, 7]],
            },
            {
                "category": "股骨",
                "segmentation": [[3, 5], [4, 5], [4, 6], [3, 6]],
            },
        ],
    }

    mask, object_count, positive_pixels = render_binary_finding_mask(
        annotation,
        {"x1": 2, "y1": 1, "x2": 9, "y2": 8},
        {"积液"},
        output_size=14,
    )
    pixels = np.asarray(mask)

    assert mask.size == (14, 14)
    assert object_count == 2
    assert positive_pixels > 0
    assert pixels[:8, :8].max() == 255
    assert pixels[8:, 8:].max() == 255
    assert set(np.unique(pixels)) <= {0, 255}


def test_lesion_guided_dataset_preserves_rgb_context_outside_binary_mask(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 8), color=(100, 120, 140)).save(image_path)
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (8, 8), color=0)
    mask.putpixel((1, 1), 255)
    mask.save(mask_path)
    record = ResearchImageRecord(
        image_key="I1",
        person_key="P1",
        diagnosis="正常",
        diagnosis_id=0,
        image_path=image_path,
        roi={"x1": 1, "y1": 0, "x2": 9, "y2": 8},
    )
    mask_record = LesionMaskRecord(
        image_key="I1",
        mask_path=mask_path,
        mask_sha256=_sha256(mask_path),
        finding_object_count=1,
        positive_pixels=1,
    )
    dataset = LesionGuidedResearchImageDataset(
        [record],
        {"I1": mask_record},
        output_size=8,
        normalize=False,
    )

    tensor = dataset[0]["image"]

    assert tensor.shape == (4, 8, 8)
    assert set(tensor[3].unique().tolist()) == {0.0, 1.0}
    assert tensor[:3, 7, 7].sum() > 0
    assert tensor[3, 7, 7] == 0


def test_mask_manifest_requires_exact_keys_and_verified_file_hashes(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (4, 4), color=0).save(mask_path)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_key",
                "mask_path",
                "mask_sha256",
                "finding_object_count",
                "positive_pixels",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_key": "I1",
                "mask_path": "mask.png",
                "mask_sha256": _sha256(mask_path),
                "finding_object_count": 0,
                "positive_pixels": 0,
            }
        )

    records = load_lesion_mask_manifest(
        manifest_path,
        tmp_path,
        _sha256(manifest_path),
        {"I1"},
    )
    assert set(records) == {"I1"}

    with pytest.raises(ValueError, match="keys differ"):
        load_lesion_mask_manifest(
            manifest_path,
            tmp_path,
            _sha256(manifest_path),
            {"I2"},
        )

    mask_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mask SHA-256 mismatch"):
        load_lesion_mask_manifest(
            manifest_path,
            tmp_path,
            _sha256(manifest_path),
            {"I1"},
        )
