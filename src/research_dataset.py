"""Patient-safe image and bag datasets backed by the frozen registry."""

from __future__ import annotations

import csv
import multiprocessing
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.research_transforms import (
    extract_region,
    letterbox_rgb,
    pil_to_imagenet_tensor,
)


@dataclass(frozen=True)
class ResearchImageRecord:
    image_key: str
    person_key: str
    diagnosis: str
    diagnosis_id: int
    image_path: Path = field(repr=False)
    roi: dict[str, float] = field(repr=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_source_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Image source escapes project root") from error
    return candidate


def load_fold_records(
    registry_dir: Path,
    project_root: Path,
    outer_fold: int,
    split: str,
) -> list[ResearchImageRecord]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    fold_rows = _read_csv(registry_dir / "folds_inner.csv")
    selected_people = {
        row["person_key"]
        for row in fold_rows
        if int(row["outer_fold"]) == outer_fold and row["split"] == split
    }
    sources = {
        row["image_key"]: row
        for row in _read_csv(registry_dir / "private" / "image_sources.csv")
    }
    records = []
    for row in _read_csv(registry_dir / "images.csv"):
        if row["include"] != "1" or row["person_key"] not in selected_people:
            continue
        source = sources.get(row["image_key"])
        if source is None:
            raise ValueError(f"Missing private source for {row['image_key']}")
        image_path = _safe_source_path(project_root, source["raw_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(
            ResearchImageRecord(
                image_key=row["image_key"],
                person_key=row["person_key"],
                diagnosis=row["diagnosis"],
                diagnosis_id=int(row["diagnosis_id"]),
                image_path=image_path,
                roi={
                    name: float(row[f"roi_{name}"])
                    for name in ("x1", "y1", "x2", "y2")
                },
            )
        )
    records.sort(key=lambda item: (item.person_key, item.image_key))
    if len({record.image_key for record in records}) != len(records):
        raise ValueError("Duplicate image_key in selected records")
    return records


class ResearchImageDataset(Dataset):
    def __init__(
        self,
        records: list[ResearchImageRecord],
        input_mode: str,
        output_size: int = 384,
        normalize: bool = True,
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
        letterbox_fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if input_mode not in {"full", "roi"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.records = list(records)
        self.input_mode = input_mode
        self.output_size = output_size
        self.normalize = normalize
        self.image_transform = image_transform
        self.letterbox_fill = letterbox_fill

    def __len__(self) -> int:
        return len(self.records)

    def load_tensor(self, record: ResearchImageRecord) -> torch.Tensor:
        with Image.open(record.image_path) as source:
            region = extract_region(source, record.roi, self.input_mode)
            resized = letterbox_rgb(region, self.output_size, fill=self.letterbox_fill)
            if self.image_transform is not None:
                resized = self.image_transform(resized)
        return pil_to_imagenet_tensor(resized, normalize=self.normalize)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": self.load_tensor(record),
            "target": record.diagnosis_id,
            "person_key": record.person_key,
            "image_key": record.image_key,
        }


def estimate_letterbox_fill(
    records: list[ResearchImageRecord],
    input_mode: str,
    max_images: int = 512,
    seed: int = 20260724,
) -> tuple[int, int, int]:
    """Estimate a neutral RGB fill from training-fold region borders only."""
    if not records:
        raise ValueError("Cannot estimate letterbox fill without images")
    if max_images < 1:
        raise ValueError("max_images must be positive")
    ordered = sorted(records, key=lambda item: item.image_key)
    if len(ordered) > max_images:
        ordered = random.Random(seed).sample(ordered, max_images)
    samples = []
    for record in ordered:
        with Image.open(record.image_path) as source:
            region = extract_region(source, record.roi, input_mode)
            pixels = np.asarray(region, dtype=np.uint8)
        border = np.concatenate(
            (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]),
            axis=0,
        )
        stride = max(1, len(border) // 512)
        samples.append(border[::stride][:512])
    median = np.median(np.concatenate(samples, axis=0), axis=0)
    return tuple(int(round(value)) for value in median)


def select_patient_instances(
    records: list[ResearchImageRecord],
    max_instances: int,
    training: bool,
    seed: int,
    epoch: int,
) -> list[ResearchImageRecord]:
    ordered = sorted(records, key=lambda item: item.image_key)
    if not training or len(ordered) <= max_instances:
        return ordered
    person_key = ordered[0].person_key
    generator = random.Random(f"{seed}:{epoch}:{person_key}")
    return sorted(
        generator.sample(ordered, max_instances),
        key=lambda item: item.image_key,
    )


class PatientBagDataset(Dataset):
    def __init__(
        self,
        image_dataset: ResearchImageDataset,
        max_instances: int = 6,
        training: bool = True,
        seed: int = 20260724,
    ) -> None:
        if max_instances <= 0:
            raise ValueError("max_instances must be positive")
        self.image_dataset = image_dataset
        self.max_instances = max_instances
        self.training = training
        self.seed = seed
        # DataLoader persistent workers keep their own Dataset copy. A shared
        # epoch counter ensures that each worker sees the new deterministic bag
        # sample without being restarted every epoch.
        self._shared_epoch = multiprocessing.Value("i", 0)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(image_dataset.records):
            groups[record.person_key].append(index)
        self.person_keys = sorted(groups)
        self.indices_by_person = groups
        patient_labels = []
        for person_key, indices in groups.items():
            labels = {
                image_dataset.records[index].diagnosis_id for index in indices
            }
            if len(labels) != 1:
                raise ValueError(f"Mixed labels for patient {person_key}")
        for person_key in self.person_keys:
            first_index = groups[person_key][0]
            patient_labels.append(image_dataset.records[first_index].diagnosis_id)
        self.labels = tuple(patient_labels)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.person_keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        person_key = self.person_keys[index]
        records = [
            self.image_dataset.records[item]
            for item in self.indices_by_person[person_key]
        ]
        selected = select_patient_instances(
            records,
            max_instances=self.max_instances,
            training=self.training,
            seed=self.seed,
            epoch=int(self._shared_epoch.value),
        )
        tensors = [self.image_dataset.load_tensor(record) for record in selected]
        return {
            "images": torch.stack(tensors),
            "target": selected[0].diagnosis_id,
            "person_key": person_key,
            "image_keys": [record.image_key for record in selected],
        }


def collate_patient_bags(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    max_instances = max(item["images"].shape[0] for item in items)
    channels, height, width = items[0]["images"].shape[1:]
    images = items[0]["images"].new_zeros(
        (batch_size, max_instances, channels, height, width)
    )
    mask = torch.zeros((batch_size, max_instances), dtype=torch.bool)
    for index, item in enumerate(items):
        count = item["images"].shape[0]
        images[index, :count] = item["images"]
        mask[index, :count] = True
    return {
        "images": images,
        "instance_mask": mask,
        "targets": torch.tensor([item["target"] for item in items], dtype=torch.long),
        "person_keys": [item["person_key"] for item in items],
        "image_keys": [item["image_keys"] for item in items],
    }
