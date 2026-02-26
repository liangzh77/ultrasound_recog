"""ISAT → COCO 格式转换。

读取清洗后的 JSON + split 文件，生成标准 COCO 格式的 annotations.json。
输出: data/coco/{train,val,test}/annotations.json + images/

用法:
    python scripts/03_convert_coco.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_category_mapping, load_isat_json, load_split_file, polygon_area, polygon_bbox, polygon_to_flat

CLEANED_DIR = ROOT / "data" / "cleaned"
SPLITS_DIR = ROOT / "data" / "splits"
COCO_DIR = ROOT / "data" / "coco"
CATEGORY_FILE = ROOT / "data" / "category_mapping.json"


def convert_split(split_name: str, rel_paths: list[str], categories: list[str], cat_to_id: dict[str, int]):
    """将一个 split 的数据转为 COCO 格式。"""
    out_dir = COCO_DIR / split_name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cat_to_id[cat], "name": cat} for cat in categories
        ],
    }

    ann_id = 1

    for img_id, rel_path in enumerate(rel_paths, start=1):
        img_path = CLEANED_DIR / rel_path
        json_path = img_path.with_suffix(".json")

        if not img_path.exists() or not json_path.exists():
            continue

        data = load_isat_json(json_path)
        info = data["info"]
        w, h = info["width"], info["height"]

        # 复制图片
        dst_img = img_dir / img_path.name
        if not dst_img.exists():
            # 处理同名图片冲突：加上疾病和患者前缀
            parts = Path(rel_path).parts
            if len(parts) >= 3:
                unique_name = f"{parts[0]}_{parts[1]}_{img_path.name}"
            else:
                unique_name = img_path.name
            dst_img = img_dir / unique_name
            shutil.copy2(img_path, dst_img)

        coco["images"].append({
            "id": img_id,
            "file_name": dst_img.name,
            "width": w,
            "height": h,
        })

        for obj in data.get("objects", []):
            cat = obj.get("category", "")
            if cat not in cat_to_id:
                continue

            seg = obj.get("segmentation", [])
            if len(seg) < 3:
                continue

            flat_seg = polygon_to_flat(seg)
            area = polygon_area(seg)
            bbox = polygon_bbox(seg)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_to_id[cat],
                "segmentation": [flat_seg],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    # 保存
    ann_path = out_dir / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    return len(coco["images"]), ann_id - 1


def main():
    if not CATEGORY_FILE.exists():
        print("错误: 类别映射文件不存在，请先运行 01_clean_labels.py")
        sys.exit(1)

    categories, cat_to_id = load_category_mapping(CATEGORY_FILE)
    print(f"类别数: {len(categories)}")

    for split_name in ("train", "val", "test"):
        split_file = SPLITS_DIR / f"{split_name}.txt"
        if not split_file.exists():
            print(f"错误: {split_file} 不存在，请先运行 02_split_dataset.py")
            sys.exit(1)

        rel_paths = load_split_file(split_file)
        n_imgs, n_anns = convert_split(split_name, rel_paths, categories, cat_to_id)
        print(f"  {split_name}: {n_imgs} images, {n_anns} annotations → {COCO_DIR / split_name}")

    print("\nCOCO 格式转换完成!")


if __name__ == "__main__":
    main()
