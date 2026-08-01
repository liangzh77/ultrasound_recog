"""ROI polygon segmentation inference and visualization.

加载 ROI polygon segmentation 最优权重，对单张图或 split 中的样本进行
`ultrasound_rect` 动态裁剪推理，并导出预测/真值叠加图。

用法:
    python tools/15_visualize_roi_poly_seg.py --split val --limit 20
    python tools/15_visualize_roi_poly_seg.py --image path\\to\\image.jpg
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (
    CATEGORY_MAPPING_FILE,
    CLEANED_DIR,
    ROI_POLY_SEG_ARTIFACTS_DIR,
    ROI_POLY_SEG_REPORTS_DIR,
    SPLITS_DIR,
    TOOLS_DIR,
)
from src.data_utils import load_category_mapping, load_isat_json, load_split_file


_CN_FONT_CACHE = {}


def get_cn_font(size: int = 20):
    if size in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[size]
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(font_path).exists():
            font = ImageFont.truetype(font_path, size)
            _CN_FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _CN_FONT_CACHE[size] = font
    return font


def import_train_module():
    script_path = TOOLS_DIR / "14_train_roi_poly_seg.py"
    spec = importlib.util.spec_from_file_location("roi_poly_seg_train", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_roi_sample(img_path: Path, json_path: Path):
    data = load_isat_json(json_path)
    roi = data.get("ultrasound_rect")
    if not roi:
        raise ValueError("missing ultrasound_rect")

    x1, y1, x2, y2 = [int(roi[k]) for k in ("x1", "y1", "x2", "y2")]
    image = Image.open(img_path).convert("RGB")
    image = image.crop((x1, y1, x2, y2))
    roi_w, roi_h = image.size

    objects = []
    for obj in data.get("objects", []):
        seg = obj.get("segmentation", [])
        if len(seg) < 3:
            continue
        shifted = []
        for px, py in seg:
            sx = min(max(float(px) - x1, 0.0), roi_w - 1)
            sy = min(max(float(py) - y1, 0.0), roi_h - 1)
            shifted.append([sx, sy])
        objects.append({
            "category": obj.get("category", ""),
            "segmentation": shifted,
        })

    return image, objects


def draw_polygon_overlay(
    image: Image.Image,
    polygons: List[Dict],
    color: Tuple[int, int, int],
    show_label: bool = True,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = get_cn_font(max(14, image.size[1] // 28))
    for poly in polygons:
        points = [tuple(p) for p in poly["segmentation"]]
        if len(points) < 3:
            continue
        draw.polygon(points, outline=color + (255,), fill=color + (60,))
        if show_label:
            x0, y0 = points[0]
            label = poly.get("category", "")
            score = poly.get("score")
            if score is not None:
                label = f"{label} {score:.2f}"
            draw.text((x0, y0), label, fill=color + (255,), font=font)
    return canvas


def masks_to_polygons(masks: np.ndarray, labels: np.ndarray, scores: np.ndarray, id_to_name: Dict[int, str], score_thresh: float):
    polygons = []
    for mask, label, score in zip(masks, labels, scores):
        if float(score) < score_thresh:
            continue
        binary = (mask > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        category = id_to_name.get(int(label) - 1, str(label))
        for contour in contours:
            contour = contour.squeeze(axis=1)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).squeeze(axis=1)
            if approx.ndim != 2 or approx.shape[0] < 3:
                approx = contour
            polygon = [[int(x), int(y)] for x, y in approx.tolist()]
            polygons.append({
                "category": category,
                "score": round(float(score), 4),
                "segmentation": polygon,
            })
    return polygons


def make_triptych(original: Image.Image, pred: Image.Image, gt: Image.Image) -> Image.Image:
    w, h = original.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(original, (0, 0))
    canvas.paste(pred, (w, 0))
    canvas.paste(gt, (w * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = get_cn_font(max(18, h // 24))
    draw.text((10, 10), "ROI视野", fill=(255, 255, 0), font=font)
    draw.text((w + 10, 10), "模型预测", fill=(255, 255, 0), font=font)
    draw.text((w * 2 + 10, 10), "人工标注", fill=(255, 255, 0), font=font)
    return canvas


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="ROI polygon segmentation visualization")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CATEGORY_MAPPING_FILE.exists():
        raise FileNotFoundError("缺少类别映射，请先重建 shared_derived")

    categories, category_to_id = load_category_mapping(CATEGORY_MAPPING_FILE)
    id_to_name = {idx: name for name, idx in category_to_id.items()}

    train_mod = import_train_module()
    device = choose_device(args.device)
    print("Device:", device)

    model = train_mod.create_model(num_classes=len(categories) + 1)
    ckpt = ROI_POLY_SEG_ARTIFACTS_DIR / "mask_rcnn" / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError("缺少 best.pth，请先训练")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    samples = []
    if args.image:
        img_path = Path(args.image)
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError("单图推理需要同名 JSON")
        samples.append((img_path, json_path))
    else:
        split_paths = load_split_file(SPLITS_DIR / f"{args.split}.txt")
        random.Random(args.seed).shuffle(split_paths)
        for rel_path in split_paths[:args.limit]:
            img_path = CLEANED_DIR / rel_path
            json_path = img_path.with_suffix(".json")
            if img_path.exists() and json_path.exists():
                samples.append((img_path, json_path))

    out_dir = ROI_POLY_SEG_REPORTS_DIR / "visualizations" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for idx, (img_path, json_path) in enumerate(samples, start=1):
            roi_image, gt_objects = load_roi_sample(img_path, json_path)
            image_tensor = train_mod.F.to_tensor(roi_image).to(device)
            outputs = model([image_tensor])[0]

            pred_polys = masks_to_polygons(
                outputs["masks"].detach().cpu().numpy()[:, 0],
                outputs["labels"].detach().cpu().numpy(),
                outputs["scores"].detach().cpu().numpy(),
                id_to_name=id_to_name,
                score_thresh=args.score_thresh,
            )

            pred_overlay = draw_polygon_overlay(roi_image, pred_polys, (255, 0, 0))
            gt_overlay = draw_polygon_overlay(roi_image, gt_objects, (0, 255, 0))
            triptych = make_triptych(roi_image, pred_overlay, gt_overlay)

            stem = img_path.stem
            save_path = out_dir / f"{idx:03d}_{stem}.png"
            triptych.save(save_path)

            manifest.append({
                "image": str(img_path),
                "json": str(json_path),
                "output": str(save_path),
                "num_predictions": len(pred_polys),
                "num_ground_truth": len(gt_objects),
            })
            print(f"[{idx}/{len(samples)}] saved: {save_path.name}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
