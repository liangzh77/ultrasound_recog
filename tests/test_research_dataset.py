import csv

import numpy as np
import torch
from PIL import Image

from src.research_dataset import (
    PatientBagDataset,
    ResearchImageDataset,
    ResearchImageRecord,
    collate_patient_bags,
    estimate_letterbox_fill,
    load_fold_records,
    select_patient_instances,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_record(tmp_path, image_key="IMG_1", person_key="PERSON_1", label=2):
    pixels = np.zeros((6, 10, 3), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0)
    pixels[1:5, 2:8] = (0, 255, 0)
    image_path = tmp_path / f"{image_key}.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    return ResearchImageRecord(
        image_key=image_key,
        person_key=person_key,
        diagnosis="测试类",
        diagnosis_id=label,
        image_path=image_path,
        roi={"x1": 2, "y1": 1, "x2": 8, "y2": 5},
    )


def test_image_dataset_returns_only_tensor_and_deidentified_keys(tmp_path):
    record = make_record(tmp_path)
    dataset = ResearchImageDataset(
        [record],
        input_mode="roi",
        output_size=8,
        normalize=False,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 8, 8)
    assert item["target"] == 2
    assert item["person_key"] == "PERSON_1"
    assert item["image_key"] == "IMG_1"
    assert "path" not in item
    assert float(item["image"][0].max()) == 0.0
    assert float(item["image"][1].max()) == 1.0


def test_optional_training_transform_is_applied_after_shared_letterbox(tmp_path):
    record = make_record(tmp_path)

    def make_blue(image):
        return Image.new("RGB", image.size, color=(0, 0, 255))

    dataset = ResearchImageDataset(
        [record],
        input_mode="roi",
        output_size=8,
        normalize=False,
        image_transform=make_blue,
    )

    tensor = dataset[0]["image"]
    assert float(tensor[0].max()) == 0.0
    assert float(tensor[1].max()) == 0.0
    assert float(tensor[2].min()) == 1.0


def test_letterbox_fill_is_estimated_from_selected_training_region_only(tmp_path):
    record = make_record(tmp_path)

    roi_fill = estimate_letterbox_fill([record], input_mode="roi")
    full_fill = estimate_letterbox_fill([record], input_mode="full")

    assert roi_fill == (0, 255, 0)
    assert full_fill != roi_fill


def test_training_instance_selection_is_limited_and_epoch_deterministic(tmp_path):
    records = [
        make_record(tmp_path, image_key=f"IMG_{index}") for index in range(10)
    ]

    first = select_patient_instances(records, 6, training=True, seed=7, epoch=2)
    repeat = select_patient_instances(records, 6, training=True, seed=7, epoch=2)
    next_epoch = select_patient_instances(records, 6, training=True, seed=7, epoch=3)
    inference = select_patient_instances(records, 6, training=False, seed=7, epoch=2)

    assert [row.image_key for row in first] == [row.image_key for row in repeat]
    assert len(first) == 6
    assert {row.image_key for row in first} != {row.image_key for row in next_epoch}
    assert len(inference) == 10


def test_variable_patient_bags_are_padded_with_a_boolean_mask(tmp_path):
    records = [
        make_record(tmp_path, image_key="A1", person_key="A", label=0),
        make_record(tmp_path, image_key="A2", person_key="A", label=0),
        make_record(tmp_path, image_key="B1", person_key="B", label=1),
    ]
    images = ResearchImageDataset(records, input_mode="roi", output_size=8)
    bags = PatientBagDataset(images, max_instances=6, training=False)

    batch = collate_patient_bags([bags[0], bags[1]])

    assert batch["images"].shape == (2, 2, 3, 8, 8)
    assert torch.equal(
        batch["instance_mask"],
        torch.tensor([[True, True], [True, False]]),
    )
    assert batch["targets"].tolist() == [0, 1]


def test_registry_loader_uses_inner_split_and_private_paths(tmp_path):
    registry = tmp_path / "registry"
    image = tmp_path / "raw" / "sample.png"
    image.parent.mkdir()
    Image.new("RGB", (4, 4)).save(image)
    write_csv(
        registry / "images.csv",
        [
            {
                "image_key": "IMG_1",
                "person_key": "PERSON_1",
                "diagnosis": "正常",
                "diagnosis_id": 0,
                "width": 4,
                "height": 4,
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 4,
                "roi_y2": 4,
                "roi_valid": 1,
                "roi_reviewed": 1,
                "annotation_object_count": 0,
                "include": 1,
            }
        ],
    )
    write_csv(
        registry / "private" / "image_sources.csv",
        [
            {
                "image_key": "IMG_1",
                "raw_image_path": image.relative_to(tmp_path).as_posix(),
                "raw_annotation_path": "raw/sample.json",
                "normalized_annotation_path": "derived/sample.json",
            }
        ],
    )
    write_csv(
        registry / "folds_inner.csv",
        [
            {
                "outer_fold": 0,
                "person_key": "PERSON_1",
                "diagnosis": "正常",
                "split": "train",
            }
        ],
    )

    records = load_fold_records(registry, tmp_path, outer_fold=0, split="train")

    assert len(records) == 1
    assert records[0].image_path == image.resolve()
    assert str(image.resolve()) not in repr(records[0])
