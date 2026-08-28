"""Class-specific manual-contour inputs for the M0--M3 patient study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.research_dataset import ResearchImageDataset, ResearchImageRecord
from src.research_transforms import letterbox_rgb, pil_to_imagenet_tensor


@dataclass(frozen=True)
class CategoryMask:
    path: Path
    sha256: str
    object_count: int
    positive_pixels: int


@dataclass(frozen=True)
class SemanticContourRecord:
    image_key: str
    category_masks: Mapping[str, CategoryMask]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roi_bounds(
    roi: Mapping[str, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, math.floor(float(roi["x1"])))
    top = max(0, math.floor(float(roi["y1"])))
    right = min(width, math.ceil(float(roi["x2"])))
    bottom = min(height, math.ceil(float(roi["y2"])))
    if left >= right or top >= bottom:
        raise ValueError("Invalid effective-view ROI")
    return left, top, right, bottom


def render_category_masks(
    annotation: Mapping[str, Any],
    roi: Mapping[str, float],
    categories: Sequence[str],
    output_size: int,
) -> tuple[dict[str, Image.Image], dict[str, int], dict[str, int]]:
    """Rasterize overlapping categories independently in one shared ROI frame."""

    if output_size <= 0:
        raise ValueError("output_size must be positive")
    category_order = tuple(str(value) for value in categories)
    if not category_order or len(category_order) != len(set(category_order)):
        raise ValueError("Semantic contour categories must be unique and nonempty")
    info = annotation.get("info") or {}
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Annotation has invalid image dimensions")
    left, top, right, bottom = _roi_bounds(roi, width, height)
    masks = {
        category: Image.new("L", (right - left, bottom - top), color=0)
        for category in category_order
    }
    draws = {category: ImageDraw.Draw(mask) for category, mask in masks.items()}
    object_counts = {category: 0 for category in category_order}
    for item in annotation.get("objects") or []:
        category = str(item.get("category", ""))
        if category not in masks:
            continue
        segmentation = item.get("segmentation") or []
        points = [
            (float(point[0]) - left, float(point[1]) - top)
            for point in segmentation
            if isinstance(point, Sequence) and len(point) >= 2
        ]
        if len(points) < 3:
            continue
        draws[category].polygon(points, fill=255)
        object_counts[category] += 1

    scale = min(output_size / (right - left), output_size / (bottom - top))
    resized_width = max(1, round((right - left) * scale))
    resized_height = max(1, round((bottom - top) * scale))
    rendered: dict[str, Image.Image] = {}
    positive_pixels: dict[str, int] = {}
    for category, mask in masks.items():
        resized = mask.resize(
            (resized_width, resized_height), resample=Image.Resampling.NEAREST
        )
        canvas = Image.new("L", (output_size, output_size), color=0)
        canvas.paste(
            resized,
            ((output_size - resized_width) // 2, (output_size - resized_height) // 2),
        )
        rendered[category] = canvas
        positive_pixels[category] = int(
            np.count_nonzero(np.asarray(canvas, dtype=np.uint8))
        )
    return rendered, object_counts, positive_pixels


def load_semantic_contour_manifest(
    manifest_path: Path,
    project_root: Path,
    expected_sha256: str,
    expected_image_keys: set[str],
    categories: Sequence[str],
    verify_mask_hashes: bool = True,
) -> dict[str, SemanticContourRecord]:
    """Load a privacy-safe sparse manifest and verify every referenced mask."""

    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != expected_sha256.casefold():
        raise ValueError("Semantic contour manifest SHA-256 mismatch")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"image_key", "category_masks_json"}.issubset(rows[0]):
        raise ValueError("Semantic contour manifest columns are incomplete")
    category_set = set(categories)
    root = project_root.resolve()
    records: dict[str, SemanticContourRecord] = {}
    for row in rows:
        image_key = row["image_key"]
        if not image_key or image_key in records:
            raise ValueError(f"Duplicate semantic contour image_key: {image_key}")
        raw_masks = json.loads(row["category_masks_json"])
        if not isinstance(raw_masks, dict) or not set(raw_masks) <= category_set:
            raise ValueError("Manifest contains an unknown semantic contour category")
        category_masks: dict[str, CategoryMask] = {}
        for category, values in raw_masks.items():
            relative = Path(str(values["path"]))
            if relative.is_absolute():
                raise ValueError("Semantic contour mask paths must be project-relative")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("Semantic contour mask path escapes project root") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_mask_sha = str(values["sha256"]).casefold()
            if verify_mask_hashes and sha256_file(path) != expected_mask_sha:
                raise ValueError(f"Semantic contour mask SHA-256 mismatch for {image_key}")
            object_count = int(values["object_count"])
            pixel_count = int(values["positive_pixels"])
            if object_count < 1 or pixel_count < 1:
                raise ValueError("Sparse semantic manifest may only list positive masks")
            category_masks[category] = CategoryMask(
                path=path,
                sha256=expected_mask_sha,
                object_count=object_count,
                positive_pixels=pixel_count,
            )
        records[image_key] = SemanticContourRecord(
            image_key=image_key,
            category_masks=category_masks,
        )
    if set(records) != expected_image_keys:
        missing = sorted(expected_image_keys - set(records))
        extra = sorted(set(records) - expected_image_keys)
        raise ValueError(
            "Semantic contour keys differ from registry; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return records


class PairedSemanticContourAugmentation:
    """Apply one sampled geometry to RGB and every semantic mask channel."""

    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(
        self, image: Image.Image, masks: Sequence[Image.Image]
    ) -> tuple[Image.Image, list[Image.Image]]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as functional

        masks = list(masks)
        if torch.rand(()) < 0.5:
            image = functional.hflip(image)
            masks = [functional.hflip(mask) for mask in masks]
        angle = float(torch.empty(()).uniform_(-7.0, 7.0))
        max_translate = round(0.03 * image.width)
        translate = [
            int(torch.randint(-max_translate, max_translate + 1, ())),
            int(torch.randint(-max_translate, max_translate + 1, ())),
        ]
        scale = float(torch.empty(()).uniform_(0.95, 1.05))
        image = functional.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=self.fill,
        )
        masks = [
            functional.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            for mask in masks
        ]
        image = functional.adjust_brightness(
            image, float(torch.empty(()).uniform_(0.85, 1.15))
        )
        image = functional.adjust_contrast(
            image, float(torch.empty(()).uniform_(0.85, 1.15))
        )
        return image, masks


class SemanticContourResearchImageDataset(ResearchImageDataset):
    """Complete effective-view RGB plus ordered, overlapping finding masks."""

    def __init__(
        self,
        records: list[ResearchImageRecord],
        contour_records: Mapping[str, SemanticContourRecord],
        categories: Sequence[str],
        output_size: int = 384,
        normalize: bool = True,
        pair_transform: PairedSemanticContourAugmentation | None = None,
        letterbox_fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(
            records,
            input_mode="roi",
            resize_mode="letterbox",
            output_size=output_size,
            normalize=normalize,
            image_transform=None,
            letterbox_fill=letterbox_fill,
        )
        if set(contour_records) != {record.image_key for record in records}:
            raise ValueError("Contour records must exactly match image records")
        self.contour_records = dict(contour_records)
        self.categories = tuple(categories)
        if not self.categories or len(self.categories) != len(set(self.categories)):
            raise ValueError("Semantic contour category order must be unique and nonempty")
        self.pair_transform = pair_transform

    def load_tensor(self, record: ResearchImageRecord) -> torch.Tensor:
        from src.research_transforms import extract_region

        with Image.open(record.image_path) as source:
            region = extract_region(source, record.roi, "roi")
            image = letterbox_rgb(region, self.output_size, fill=self.letterbox_fill)
        contour = self.contour_records[record.image_key]
        masks: list[Image.Image] = []
        for category in self.categories:
            category_mask = contour.category_masks.get(category)
            if category_mask is None:
                mask = Image.new("L", (self.output_size, self.output_size), color=0)
            else:
                with Image.open(category_mask.path) as source_mask:
                    mask = source_mask.convert("L").copy()
                if mask.size != (self.output_size, self.output_size):
                    raise ValueError(f"Unexpected semantic mask size for {record.image_key}")
            masks.append(mask)
        if self.pair_transform is not None:
            image, masks = self.pair_transform(image, masks)
        rgb_tensor = pil_to_imagenet_tensor(image, normalize=self.normalize)
        mask_tensors = []
        for mask in masks:
            pixels = np.asarray(mask, dtype=np.float32).copy()
            tensor = torch.from_numpy(pixels).div_(255.0)
            mask_tensors.append((tensor >= 0.5).to(rgb_tensor.dtype))
        return torch.cat((rgb_tensor, torch.stack(mask_tensors, dim=0)), dim=0)
