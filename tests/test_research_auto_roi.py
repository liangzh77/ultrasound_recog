from pathlib import Path

from PIL import Image

from src.research_auto_roi import ROIBox, materialize_yolo_dataset


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

