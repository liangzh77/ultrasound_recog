"""ISAT → nnU-Net v2 格式转换。

将多边形标注转为像素级 PNG 掩码，组织为 nnU-Net v2 目录结构。
输出: data/nnunet/Dataset001_Knee/{imagesTr,labelsTr,imagesTs,labelsTs}/

用法:
    python scripts/05_convert_nnunet.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_category_mapping, load_isat_json, load_split_file

CLEANED_DIR = ROOT / "data" / "cleaned"
SPLITS_DIR = ROOT / "data" / "splits"
NNUNET_DIR = ROOT / "data" / "nnunet" / "Dataset001_Knee"
CATEGORY_FILE = ROOT / "data" / "category_mapping.json"


def create_segmentation_mask(json_data: dict, cat_to_id: dict[str, int], width: int, height: int) -> np.ndarray:
    """从 ISAT 标注创建像素级分割掩码。"""
    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in json_data.get("objects", []):
        cat = obj.get("category", "")
        if cat not in cat_to_id:
            continue

        seg = obj.get("segmentation", [])
        if len(seg) < 3:
            continue

        # nnU-Net 标签值从 1 开始（0 是背景）
        label_val = cat_to_id[cat] + 1

        # 转为 (x, y) 元组列表供 PIL 绘制
        polygon = [(p[0], p[1]) for p in seg]

        # 绘制填充多边形
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.polygon(polygon, fill=int(label_val))
        mask = np.array(img)

    return mask


def convert_split(split_name: str, rel_paths: list[str], cat_to_id: dict[str, int],
                  is_train: bool = True):
    """转换一个 split 的数据为 nnU-Net 格式。"""
    if is_train:
        img_dir = NNUNET_DIR / "imagesTr"
        lbl_dir = NNUNET_DIR / "labelsTr"
    else:
        img_dir = NNUNET_DIR / "imagesTs"
        lbl_dir = NNUNET_DIR / "labelsTs"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, rel_path in enumerate(rel_paths):
        img_path = CLEANED_DIR / rel_path
        json_path = img_path.with_suffix(".json")

        if not img_path.exists() or not json_path.exists():
            continue

        data = load_isat_json(json_path)
        info = data["info"]
        w, h = info["width"], info["height"]

        # nnU-Net 文件命名: case_XXXXX_0000.png (通道编号)
        case_id = f"case_{count:05d}"

        # 复制图片（nnU-Net 要求 _0000 后缀表示通道）
        dst_img = img_dir / f"{case_id}_0000.png"
        if img_path.suffix.lower() in (".jpg", ".jpeg"):
            # 转为 PNG
            im = Image.open(img_path)
            im.save(dst_img)
        else:
            shutil.copy2(img_path, dst_img)

        # 生成掩码
        mask = create_segmentation_mask(data, cat_to_id, w, h)
        mask_img = Image.fromarray(mask)
        mask_img.save(lbl_dir / f"{case_id}.png")

        count += 1

    return count


def generate_dataset_json(categories: list[str], n_train: int, n_test: int):
    """生成 nnU-Net v2 的 dataset.json。"""
    # 标签映射：0 是背景，1~ 是各类别
    labels = {"background": 0}
    for i, cat in enumerate(categories):
        labels[cat] = i + 1

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": labels,
        "numTraining": n_train,
        "file_ending": ".png",
    }

    with open(NNUNET_DIR / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)


def main():
    if not CATEGORY_FILE.exists():
        print("错误: 类别映射文件不存在，请先运行 01_clean_labels.py")
        sys.exit(1)

    categories, cat_to_id = load_category_mapping(CATEGORY_FILE)
    print(f"类别数: {len(categories)}")

    # train + val 合并为 nnU-Net 的 Tr（nnU-Net 自己做交叉验证）
    train_paths = load_split_file(SPLITS_DIR / "train.txt")
    val_paths = load_split_file(SPLITS_DIR / "val.txt")
    test_paths = load_split_file(SPLITS_DIR / "test.txt")

    tr_paths = train_paths + val_paths

    print("转换训练集 (train + val)...")
    n_train = convert_split("train", tr_paths, cat_to_id, is_train=True)
    print(f"  训练样本: {n_train}")

    print("转换测试集...")
    n_test = convert_split("test", test_paths, cat_to_id, is_train=False)
    print(f"  测试样本: {n_test}")

    generate_dataset_json(categories, n_train, n_test)
    print(f"\ndataset.json 已生成: {NNUNET_DIR / 'dataset.json'}")
    print("nnU-Net 格式转换完成!")


if __name__ == "__main__":
    main()
