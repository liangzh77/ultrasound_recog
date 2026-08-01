"""Image transforms shared by full-image and ROI research experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def extract_region(
    image: Image.Image,
    roi: Mapping[str, float] | None,
    input_mode: str,
) -> Image.Image:
    if input_mode == "full":
        return image.convert("RGB").copy()
    if input_mode != "roi":
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    if roi is None:
        raise ValueError("ROI mode requires a rectangle")

    left = max(0, math.floor(float(roi["x1"])))
    top = max(0, math.floor(float(roi["y1"])))
    right = min(image.width, math.ceil(float(roi["x2"])))
    bottom = min(image.height, math.ceil(float(roi["y2"])))
    if left >= right or top >= bottom:
        raise ValueError(f"Invalid ROI after clamping: {(left, top, right, bottom)}")
    return image.crop((left, top, right, bottom)).convert("RGB")


def letterbox_rgb(
    image: Image.Image,
    output_size: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    image = image.convert("RGB")
    scale = min(output_size / image.width, output_size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (output_size, output_size), color=fill)
    offset = (
        (output_size - resized_width) // 2,
        (output_size - resized_height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def pil_to_imagenet_tensor(
    image: Image.Image,
    normalize: bool = True,
) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).div_(255.0)
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def build_training_augmentation(fill: tuple[int, int, int] = (0, 0, 0)):
    """Mild geometry/intensity transforms that preserve Doppler color."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=fill,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    )
