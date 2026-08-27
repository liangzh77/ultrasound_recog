"""Manual-lesion-mask inputs for patient-level diagnosis research."""

from __future__ import annotations

import csv
import hashlib
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
class LesionMaskRecord:
    image_key: str
    mask_path: Path
    mask_sha256: str
    finding_object_count: int
    positive_pixels: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roi_pixel_bounds(
    roi: Mapping[str, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, math.floor(float(roi["x1"])))
    top = max(0, math.floor(float(roi["y1"])))
    right = min(width, math.ceil(float(roi["x2"])))
    bottom = min(height, math.ceil(float(roi["y2"])))
    if left >= right or top >= bottom:
        raise ValueError("Invalid effective-view ROI")
    return left, top, right, bottom


def render_binary_finding_mask(
    annotation: Mapping[str, Any],
    roi: Mapping[str, float],
    included_categories: set[str],
    output_size: int,
) -> tuple[Image.Image, int, int]:
    """Rasterize finding polygons in the complete effective-view coordinate frame."""

    if output_size <= 0:
        raise ValueError("output_size must be positive")
    info = annotation.get("info") or {}
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Annotation has invalid image dimensions")
    left, top, right, bottom = roi_pixel_bounds(roi, width, height)
    mask = Image.new("L", (right - left, bottom - top), color=0)
    draw = ImageDraw.Draw(mask)
    finding_object_count = 0
    for item in annotation.get("objects") or []:
        if str(item.get("category")) not in included_categories:
            continue
        segmentation = item.get("segmentation") or []
        if len(segmentation) < 3:
            continue
        points = []
        for point in segmentation:
            if not isinstance(point, Sequence) or len(point) < 2:
                continue
            points.append((float(point[0]) - left, float(point[1]) - top))
        if len(points) < 3:
            continue
        draw.polygon(points, fill=255)
        finding_object_count += 1

    scale = min(output_size / mask.width, output_size / mask.height)
    resized_width = max(1, round(mask.width * scale))
    resized_height = max(1, round(mask.height * scale))
    resized = mask.resize(
        (resized_width, resized_height), resample=Image.Resampling.NEAREST
    )
    canvas = Image.new("L", (output_size, output_size), color=0)
    offset = (
        (output_size - resized_width) // 2,
        (output_size - resized_height) // 2,
    )
    canvas.paste(resized, offset)
    positive_pixels = int(np.count_nonzero(np.asarray(canvas, dtype=np.uint8)))
    return canvas, finding_object_count, positive_pixels


def load_lesion_mask_manifest(
    manifest_path: Path,
    project_root: Path,
    expected_sha256: str,
    expected_image_keys: set[str],
    verify_mask_hashes: bool = True,
) -> dict[str, LesionMaskRecord]:
    manifest_path = manifest_path.resolve()
    if _sha256(manifest_path) != expected_sha256.casefold():
        raise ValueError("Lesion mask manifest SHA-256 mismatch")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_key",
        "mask_path",
        "mask_sha256",
        "finding_object_count",
        "positive_pixels",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Lesion mask manifest columns are incomplete")
    root = project_root.resolve()
    records: dict[str, LesionMaskRecord] = {}
    for row in rows:
        image_key = row["image_key"]
        if image_key in records:
            raise ValueError(f"Duplicate lesion mask image_key: {image_key}")
        relative = Path(row["mask_path"])
        if relative.is_absolute():
            raise ValueError("Lesion mask paths must be project-relative")
        mask_path = (root / relative).resolve()
        try:
            mask_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Lesion mask path escapes project root") from error
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask_hash = row["mask_sha256"].casefold()
        if verify_mask_hashes and _sha256(mask_path) != mask_hash:
            raise ValueError(f"Lesion mask SHA-256 mismatch for {image_key}")
        records[image_key] = LesionMaskRecord(
            image_key=image_key,
            mask_path=mask_path,
            mask_sha256=mask_hash,
            finding_object_count=int(row["finding_object_count"]),
            positive_pixels=int(row["positive_pixels"]),
        )
    if set(records) != expected_image_keys:
        missing = sorted(expected_image_keys - set(records))
        extra = sorted(set(records) - expected_image_keys)
        raise ValueError(
            f"Lesion mask keys differ from registry; missing={missing[:3]}, extra={extra[:3]}"
        )
    return records


class PairedLesionAugmentation:
    """Apply identical geometry to RGB and mask; intensity changes affect RGB only."""

    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as functional

        if torch.rand(()) < 0.5:
            image = functional.hflip(image)
            mask = functional.hflip(mask)
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
        mask = functional.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        image = functional.adjust_brightness(
            image, float(torch.empty(()).uniform_(0.85, 1.15))
        )
        image = functional.adjust_contrast(
            image, float(torch.empty(()).uniform_(0.85, 1.15))
        )
        return image, mask


class LesionGuidedResearchImageDataset(ResearchImageDataset):
    """Complete effective-view RGB plus a co-registered binary finding channel."""

    def __init__(
        self,
        records: list[ResearchImageRecord],
        mask_records: Mapping[str, LesionMaskRecord],
        output_size: int = 384,
        normalize: bool = True,
        pair_transform: PairedLesionAugmentation | None = None,
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
        if set(mask_records) != {record.image_key for record in records}:
            raise ValueError("Mask records must exactly match image records")
        self.mask_records = dict(mask_records)
        self.pair_transform = pair_transform

    def load_tensor(self, record: ResearchImageRecord) -> torch.Tensor:
        from src.research_transforms import extract_region

        with Image.open(record.image_path) as source:
            region = extract_region(source, record.roi, "roi")
            image = letterbox_rgb(region, self.output_size, fill=self.letterbox_fill)
        with Image.open(self.mask_records[record.image_key].mask_path) as source_mask:
            mask = source_mask.convert("L").copy()
        if mask.size != (self.output_size, self.output_size):
            raise ValueError(f"Unexpected lesion mask size for {record.image_key}")
        if self.pair_transform is not None:
            image, mask = self.pair_transform(image, mask)
        rgb_tensor = pil_to_imagenet_tensor(image, normalize=self.normalize)
        mask_pixels = np.asarray(mask, dtype=np.float32).copy()
        mask_tensor = torch.from_numpy(mask_pixels).unsqueeze(0).div_(255.0)
        mask_tensor = (mask_tensor >= 0.5).to(rgb_tensor.dtype)
        return torch.cat((rgb_tensor, mask_tensor), dim=0)
