"""ISAT → MedSAM 格式转换。

将每个标注对象转为一个二值掩码 PNG + 对应的 bbox prompt。
输出: workspace/data/shared_derived/medsam/{train,val,test}/{images,masks}/ + bboxes.json

用法:
    python tools/06_convert_medsam.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import CATEGORY_MAPPING_FILE, CLEANED_DIR, MEDSAM_DIR, SPLITS_DIR
from src.data_utils import load_category_mapping, load_isat_json, load_split_file

CATEGORY_FILE = CATEGORY_MAPPING_FILE


def convert_split(split_name: str, rel_paths: list[str], cat_to_id: dict[str, int]):
    """转换一个 split 的数据为 MedSAM 格式。"""
    img_dir = MEDSAM_DIR / split_name / "images"
    mask_dir = MEDSAM_DIR / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    bbox_entries = []
    sample_idx = 0

    for rel_path in rel_paths:
        img_path = CLEANED_DIR / rel_path
        json_path = img_path.with_suffix(".json")

        if not img_path.exists() or not json_path.exists():
            continue

        data = load_isat_json(json_path)
        info = data["info"]
        w, h = info["width"], info["height"]

        objects = data.get("objects", [])
        if not objects:
            continue

        # 复制原图（转 PNG）
        case_id = f"sample_{sample_idx:05d}"
        dst_img = img_dir / f"{case_id}.png"
        im = Image.open(img_path).convert("RGB")
        im.save(dst_img)

        # 为每个标注对象生成独立的二值掩码
        for obj_idx, obj in enumerate(objects):
            cat = obj.get("category", "")
            if cat not in cat_to_id:
                continue

            seg = obj.get("segmentation", [])
            if len(seg) < 3:
                continue

            # 创建二值掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = [(p[0], p[1]) for p in seg]
            mask_img = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(polygon, fill=255)
            mask = np.array(mask_img)

            # 计算 bbox [x_min, y_min, x_max, y_max]
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            # 保存掩码
            mask_name = f"{case_id}_obj{obj_idx:03d}.png"
            Image.fromarray(mask).save(mask_dir / mask_name)

            bbox_entries.append({
                "image": f"{case_id}.png",
                "mask": mask_name,
                "bbox": bbox,
                "category": cat,
                "category_id": cat_to_id[cat],
                "width": w,
                "height": h,
            })

        sample_idx += 1

    # 保存 bbox 信息
    bbox_path = MEDSAM_DIR / split_name / "bboxes.json"
    with open(bbox_path, "w", encoding="utf-8") as f:
        json.dump(bbox_entries, f, ensure_ascii=False, indent=2)

    return sample_idx, len(bbox_entries)


def main():
    if not CATEGORY_FILE.exists():
        print("错误: 类别映射文件不存在，请先运行 01_clean_labels.py")
        sys.exit(1)

    categories, cat_to_id = load_category_mapping(CATEGORY_FILE)
    print(f"类别数: {len(categories)}")

    for split in ("train", "val", "test"):
        split_file = SPLITS_DIR / f"{split}.txt"
        rel_paths = load_split_file(split_file)
        n_imgs, n_masks = convert_split(split, rel_paths, cat_to_id)
        print(f"  {split}: {n_imgs} images, {n_masks} mask-bbox pairs")

    print("\nMedSAM 格式转换完成!")


if __name__ == "__main__":
    main()
