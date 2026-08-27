"""Contracts and dataset preparation for automatic ultrasound-view ROI research."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml
import numpy as np


@dataclass(frozen=True)
class ROIBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, object], prefix: str = "roi_") -> "ROIBox":
        return cls(*(float(row[f"{prefix}{name}"]) for name in ("x1", "y1", "x2", "y2")))

    def validate(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if not (0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height):
            raise ValueError(f"ROI outside image bounds: {self} for {width}x{height}")

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        self.validate(width, height)
        return (
            (self.x1 + self.x2) / (2 * width),
            (self.y1 + self.y2) / (2 * height),
            (self.x2 - self.x1) / width,
            (self.y2 - self.y1) / height,
        )

    def iou(self, other: "ROIBox") -> float:
        intersection_width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        intersection_height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = intersection_width * intersection_height
        self_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        other_area = (other.x2 - other.x1) * (other.y2 - other.y1)
        union = self_area + other_area - intersection
        return intersection / union if union > 0 else 0.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptive_distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("descriptive values must be finite")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def load_r0_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("experiment_code") != "R0":
        raise ValueError("R0 configuration must declare experiment_code: R0")
    return payload


def _link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing target differs in size: {target}")
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def materialize_yolo_dataset(
    *,
    root: Path,
    image_rows: Iterable[Mapping[str, str]],
    source_by_image: Mapping[str, str],
    split_by_person: Mapping[str, str],
    output_dir: Path,
) -> dict:
    """Create a privacy-safe one-class YOLO dataset using stable image keys."""
    split_alias = {"train": "train", "validation": "val", "test": "test"}
    counts = {"train": 0, "val": 0, "test": 0}
    link_modes: dict[str, int] = {}
    public_rows: list[dict[str, str]] = []

    for row in image_rows:
        if row.get("include") != "1":
            continue
        image_key = row["image_key"]
        person_key = row["person_key"]
        try:
            split = split_alias[split_by_person[person_key]]
            raw_relative = source_by_image[image_key]
        except KeyError as error:
            raise ValueError(f"missing R0 mapping for {error.args[0]}") from error

        source = (root / raw_relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        width, height = int(row["width"]), int(row["height"])
        box = ROIBox.from_mapping(row)
        normalized = box.to_yolo(width, height)

        image_target = output_dir / "images" / split / f"{image_key}{source.suffix.lower()}"
        label_target = output_dir / "labels" / split / f"{image_key}.txt"
        mode = _link_or_copy(source, image_target)
        link_modes[mode] = link_modes.get(mode, 0) + 1
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(
            "0 " + " ".join(f"{value:.10f}" for value in normalized) + "\n",
            encoding="utf-8",
        )
        counts[split] += 1
        public_rows.append(
            {
                "image_key": image_key,
                "person_key": person_key,
                "split": split,
                "width": str(width),
                "height": str(height),
                "roi_x1": row["roi_x1"],
                "roi_y1": row["roi_y1"],
                "roi_x2": row["roi_x2"],
                "roi_y2": row["roi_y2"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "ultrasound_view"},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(public_rows[0]) if public_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)

    summary = {
        "images": sum(counts.values()),
        "split_counts": counts,
        "materialization": link_modes,
        "contains_raw_paths": False,
        "geometry_metrics_are_descriptive": True,
        "primary_decision_basis": "downstream_patient_classification",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
