"""患者级数据集拆分脚本。

按患者维度进行分层抽样，保证：
1. 同一患者的所有图片只出现在同一集合中
2. 每个疾病类别内独立拆分，保持类别分布一致
3. 比例: train 70% / val 15% / test 15%

用法:
    python scripts/02_split_dataset.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
SPLITS_DIR = ROOT / "data" / "splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def collect_patient_samples(data_dir: Path):
    """收集每个 (疾病, 患者) 下的所有样本路径。"""
    disease_patients = defaultdict(lambda: defaultdict(list))

    for disease_dir in sorted(data_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name

        for patient_dir in sorted(disease_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient = patient_dir.name

            for img_file in sorted(patient_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    json_file = img_file.with_suffix(".json")
                    if json_file.exists():
                        # 存储相对于 data_dir 的路径
                        rel_path = img_file.relative_to(data_dir)
                        disease_patients[disease][patient].append(str(rel_path))

    return disease_patients


def stratified_patient_split(disease_patients, seed=42):
    """分层抽样：每个疾病类别内独立按患者拆分。"""
    rng = np.random.RandomState(seed)

    train_files, val_files, test_files = [], [], []
    stats = {}

    for disease in sorted(disease_patients.keys()):
        patients = sorted(disease_patients[disease].keys())
        rng.shuffle(patients)

        n = len(patients)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # test 取剩余
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train -= 1
            n_test = 1

        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]

        train_imgs, val_imgs, test_imgs = [], [], []
        for p in train_patients:
            train_imgs.extend(disease_patients[disease][p])
        for p in val_patients:
            val_imgs.extend(disease_patients[disease][p])
        for p in test_patients:
            test_imgs.extend(disease_patients[disease][p])

        train_files.extend(train_imgs)
        val_files.extend(val_imgs)
        test_files.extend(test_imgs)

        stats[disease] = {
            "patients": {"train": len(train_patients), "val": len(val_patients), "test": len(test_patients), "total": n},
            "images": {"train": len(train_imgs), "val": len(val_imgs), "test": len(test_imgs), "total": len(train_imgs) + len(val_imgs) + len(test_imgs)},
        }

    return train_files, val_files, test_files, stats


def main():
    print(f"数据目录: {CLEANED_DIR}")

    if not CLEANED_DIR.exists():
        print("错误: 清洗后的数据目录不存在，请先运行 01_clean_labels.py")
        sys.exit(1)

    disease_patients = collect_patient_samples(CLEANED_DIR)
    total_patients = sum(len(ps) for ps in disease_patients.values())
    total_images = sum(
        len(imgs) for ps in disease_patients.values() for imgs in ps.values()
    )
    print(f"  疾病类别: {len(disease_patients)}")
    print(f"  总患者数: {total_patients}")
    print(f"  总样本数: {total_images}")

    train, val, test, stats = stratified_patient_split(disease_patients, RANDOM_SEED)

    print(f"\n拆分结果:")
    print(f"  Train: {len(train)} images")
    print(f"  Val:   {len(val)} images")
    print(f"  Test:  {len(test)} images")

    # 保存
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, files in [("train", train), ("val", val), ("test", test)]:
        path = SPLITS_DIR / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for line in sorted(files):
                f.write(line + "\n")
        print(f"  已保存: {path} ({len(files)} 条)")

    # 保存统计信息
    split_stats = {
        "seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "totals": {"train": len(train), "val": len(val), "test": len(test)},
        "per_disease": stats,
    }
    stats_path = SPLITS_DIR / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)
    print(f"  统计信息: {stats_path}")

    # 打印各类别详情
    print("\n各疾病类别拆分详情:")
    print(f"  {'疾病':<12} {'患者(tr/va/te)':<18} {'图片(tr/va/te)':<20}")
    print("  " + "-" * 50)
    for disease, s in sorted(stats.items()):
        p = s["patients"]
        i = s["images"]
        print(f"  {disease:<12} {p['train']:>3}/{p['val']:>3}/{p['test']:>3}        {i['train']:>4}/{i['val']:>4}/{i['test']:>4}")


if __name__ == "__main__":
    main()
