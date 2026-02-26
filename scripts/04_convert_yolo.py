"""COCO → YOLO-seg 格式转换。

从 COCO annotations.json 转为 YOLO 分割标签格式。
每个图片一个 .txt，每行: class_id x1 y1 x2 y2 ... (归一化坐标)
同时生成 configs/yolo_seg.yaml 配置文件。

用法:
    python scripts/04_convert_yolo.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_category_mapping

COCO_DIR = ROOT / "data" / "coco"
YOLO_DIR = ROOT / "data" / "yolo_seg"
CATEGORY_FILE = ROOT / "data" / "category_mapping.json"
CONFIG_DIR = ROOT / "configs"


def convert_coco_to_yolo(split_name: str, categories: list[str], cat_to_id: dict[str, int]):
    """将一个 split 的 COCO 格式转为 YOLO-seg 格式。"""
    coco_split = COCO_DIR / split_name
    ann_path = coco_split / "annotations.json"

    if not ann_path.exists():
        print(f"  跳过 {split_name}: annotations.json 不存在")
        return 0

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    # 构建 image_id → image_info 映射
    id_to_img = {img["id"]: img for img in coco["images"]}

    # 构建 image_id → annotations 映射
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # 输出目录
    img_out = YOLO_DIR / "images" / split_name
    lbl_out = YOLO_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, img_info in id_to_img.items():
        w = img_info["width"]
        h = img_info["height"]
        fname = img_info["file_name"]

        # 复制图片
        src_img = coco_split / "images" / fname
        dst_img = img_out / fname
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # 生成标签文件
        label_lines = []
        for ann in img_anns.get(img_id, []):
            cls_id = ann["category_id"]
            seg = ann["segmentation"]
            if not seg or not seg[0]:
                continue

            # COCO segmentation 是 [x1,y1,x2,y2,...] 扁平列表
            flat = seg[0]
            # 归一化坐标
            normalized = []
            for i in range(0, len(flat), 2):
                nx = flat[i] / w
                ny = flat[i + 1] / h
                # 裁剪到 [0, 1]
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                normalized.extend([nx, ny])

            coords_str = " ".join(f"{v:.6f}" for v in normalized)
            label_lines.append(f"{cls_id} {coords_str}")

        # 写标签文件
        stem = Path(fname).stem
        lbl_path = lbl_out / f"{stem}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(label_lines))
        count += 1

    return count


def generate_yaml(categories: list[str], cat_to_id: dict[str, int]):
    """生成 YOLO 数据集配置文件。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    yaml_content = f"# 膝关节超声 YOLO-seg 数据集配置\n"
    yaml_content += f"path: {YOLO_DIR.resolve()}\n"
    yaml_content += f"train: images/train\n"
    yaml_content += f"val: images/val\n"
    yaml_content += f"test: images/test\n\n"
    yaml_content += f"# 类别数\n"
    yaml_content += f"nc: {len(categories)}\n\n"
    yaml_content += f"# 类别名称\n"
    yaml_content += f"names:\n"
    for cat in categories:
        idx = cat_to_id[cat]
        yaml_content += f"  {idx}: {cat}\n"

    yaml_path = CONFIG_DIR / "yolo_seg.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    if not CATEGORY_FILE.exists():
        print("错误: 类别映射文件不存在，请先运行 01_clean_labels.py")
        sys.exit(1)

    categories, cat_to_id = load_category_mapping(CATEGORY_FILE)
    print(f"类别数: {len(categories)}")

    for split in ("train", "val", "test"):
        count = convert_coco_to_yolo(split, categories, cat_to_id)
        print(f"  {split}: {count} 个标签文件")

    yaml_path = generate_yaml(categories, cat_to_id)
    print(f"\nYOLO 配置已生成: {yaml_path}")
    print("YOLO-seg 格式转换完成!")


if __name__ == "__main__":
    main()
