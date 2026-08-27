from pathlib import Path

from PIL import Image

from src.research_dataset import ResearchImageRecord, replace_record_rois
from src.research_auto_roi import (
    ROIBox,
    descriptive_distribution,
    materialize_yolo_dataset,
    require_sha256,
    sha256_file,
)


def test_roi_box_yolo_conversion_and_iou():
    box = ROIBox(20, 10, 80, 50)
    assert box.to_yolo(100, 100) == (0.5, 0.3, 0.6, 0.4)
    assert box.iou(box) == 1.0
    assert box.iou(ROIBox(80, 50, 90, 60)) == 0.0


def test_materialized_dataset_uses_public_keys_and_patient_splits(tmp_path: Path):
    source = tmp_path / "private_name.jpg"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(source)
    rows = [
        {
            "image_key": "IMG_PUBLIC_1",
            "person_key": "PERSON_PUBLIC_1",
            "width": "100",
            "height": "80",
            "roi_x1": "10",
            "roi_y1": "8",
            "roi_x2": "90",
            "roi_y2": "72",
            "include": "1",
        }
    ]
    output = tmp_path / "dataset"
    summary = materialize_yolo_dataset(
        root=tmp_path,
        image_rows=rows,
        source_by_image={"IMG_PUBLIC_1": source.name},
        split_by_person={"PERSON_PUBLIC_1": "validation"},
        output_dir=output,
    )

    assert summary["split_counts"] == {"train": 0, "val": 1, "test": 0}
    assert (output / "images/val/IMG_PUBLIC_1.jpg").is_file()
    assert (output / "labels/val/IMG_PUBLIC_1.txt").read_text(encoding="utf-8").startswith("0 ")
    manifest = (output / "manifest.csv").read_text(encoding="utf-8")
    assert "private_name" not in manifest
    assert "IMG_PUBLIC_1" in manifest


def test_sha256_file_is_stable(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"automatic-roi")
    assert sha256_file(artifact) == "8d0d38c12984170166adcc404cc4946eb886080c893df064303ef7721cfaf915"


def test_require_sha256_rejects_changed_frozen_artifact(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    expected = sha256_file(artifact)
    assert require_sha256(artifact, expected.upper(), "test artifact") == expected
    artifact.write_bytes(b"changed")
    try:
        require_sha256(artifact, expected, "test artifact")
    except RuntimeError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Changed frozen artifact was accepted")


def test_descriptive_distribution_keeps_continuous_geometry_without_gate():
    summary = descriptive_distribution([0.2, 0.4, 0.6, 0.8])
    assert summary["n"] == 4
    assert summary["mean"] == 0.5
    assert summary["median"] == 0.5
    assert "passed" not in summary
    assert descriptive_distribution([])["median"] is None


def test_replace_record_rois_supports_detected_and_full_image_fallback(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 80)).save(image_path)
    records = [
        ResearchImageRecord("A", "P1", "正常", 0, image_path, {"x1": 1, "y1": 1, "x2": 2, "y2": 2}),
        ResearchImageRecord("B", "P1", "正常", 0, image_path, {"x1": 1, "y1": 1, "x2": 2, "y2": 2}),
    ]
    updated, fallback = replace_record_rois(
        records,
        {
            "A": {"x1": 10, "y1": 8, "x2": 90, "y2": 72},
            "B": None,
        },
    )
    assert updated[0].roi == {"x1": 10.0, "y1": 8.0, "x2": 90.0, "y2": 72.0}
    assert updated[1].roi == {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 80.0}
    assert fallback == ("B",)
