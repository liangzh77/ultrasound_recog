import numpy as np
import torch
from PIL import Image

from src.research_transforms import (
    extract_region,
    letterbox_rgb,
    pil_to_imagenet_tensor,
)


def test_roi_mode_excludes_every_outside_pixel_without_modifying_source():
    pixels = np.zeros((6, 10, 3), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0)
    pixels[1:5, 2:8] = (0, 255, 0)
    source = Image.fromarray(pixels, mode="RGB")

    roi = extract_region(
        source,
        {"x1": 2, "y1": 1, "x2": 8, "y2": 5},
        input_mode="roi",
    )

    assert roi.size == (6, 4)
    assert set(map(tuple, np.asarray(roi).reshape(-1, 3))) == {(0, 255, 0)}
    assert tuple(np.asarray(source)[0, 0]) == (255, 0, 0)


def test_full_mode_keeps_entire_image_and_rejects_unknown_mode():
    source = Image.new("RGB", (10, 6), color=(1, 2, 3))

    assert extract_region(source, None, input_mode="full").size == (10, 6)

    try:
        extract_region(source, None, input_mode="unknown")
    except ValueError as error:
        assert "input_mode" in str(error)
    else:
        raise AssertionError("unknown input mode must fail")


def test_letterbox_preserves_aspect_ratio_and_centers_content():
    source = Image.new("RGB", (10, 4), color=(20, 40, 60))

    output = letterbox_rgb(source, output_size=10, fill=(0, 0, 0))
    pixels = np.asarray(output)

    assert output.size == (10, 10)
    assert np.all(pixels[:3] == 0)
    assert np.all(pixels[3:7] == (20, 40, 60))
    assert np.all(pixels[7:] == 0)


def test_tensor_conversion_preserves_rgb_channels_before_normalization():
    source = Image.new("RGB", (2, 2), color=(255, 128, 0))

    tensor = pil_to_imagenet_tensor(source, normalize=False)

    assert tensor.shape == (3, 2, 2)
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))
