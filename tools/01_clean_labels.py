"""标签清洗脚本。

遍历 workspace/data/raw/膝关节已标注/ 下所有 JSON 标注文件：
1. 应用 label_mapping.py 中的修复规则
2. 输出清洗后的数据到 workspace/data/shared_derived/cleaned/（保持原目录结构）
3. 图片通过软链接/复制关联
4. 生成清洗报告 workspace/data/shared_derived/clean_report.json

用法:
    python tools/01_clean_labels.py [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import (
    LABEL_FIX_MAP,
    ORPHAN_LABELS,
    SUSPICIOUS_LABELS,
    fix_label,
    get_disease_from_label,
)
from src.common_paths import (
    CATEGORY_MAPPING_FILE,
    CLEANED_DIR,
    CLEAN_REPORT_FILE,
    RAW_LABEL_DIR,
)

RAW_DIR = RAW_LABEL_DIR


def find_all_samples(raw_dir: Path):
    """扫描所有样本，返回 (disease, patient, stem, json_path, img_path) 列表。"""
    samples = []
    orphan_jsons = []
    missing_jsons = []

    for disease_dir in sorted(raw_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name

        for patient_dir in sorted(disease_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient = patient_dir.name

            # 收集该患者目录下所有 json 和图片
            jsons = {f.stem: f for f in patient_dir.glob("*.json")}
            images = {}
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for f in patient_dir.glob(ext):
                    images[f.stem] = f

            # 配对
            all_stems = set(jsons.keys()) | set(images.keys())
            for stem in sorted(all_stems):
                jp = jsons.get(stem)
                ip = images.get(stem)
                if jp and ip:
                    samples.append((disease, patient, stem, jp, ip))
                elif jp and not ip:
                    orphan_jsons.append(str(jp))
                elif ip and not jp:
                    missing_jsons.append(str(ip))

    return samples, orphan_jsons, missing_jsons


def clean_annotations(json_path: Path, disease: str):
    """清洗单个 JSON 文件的标注，返回 (cleaned_data, changes)。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    changes = []
    for obj in data.get("objects", []):
        old_cat = obj.get("category", "")
        new_cat = fix_label(old_cat, disease)

        if new_cat != old_cat:
            changes.append({"old": old_cat, "new": new_cat})
            obj["category"] = new_cat

    return data, changes


def main():
    parser = argparse.ArgumentParser(description="标签清洗")
    parser.add_argument("--dry-run", action="store_true", help="仅生成报告，不输出文件")
    args = parser.parse_args()

    print(f"扫描数据目录: {RAW_DIR}")
    samples, orphan_jsons, missing_jsons = find_all_samples(RAW_DIR)
    print(f"  配对样本: {len(samples)}")
    print(f"  孤立 JSON（无图片）: {len(orphan_jsons)}")
    print(f"  缺标注图片: {len(missing_jsons)}")

    # 清洗
    report = {
        "total_samples": len(samples),
        "orphan_jsons": orphan_jsons,
        "missing_json_images": missing_jsons,
        "label_changes": [],
        "suspicious_labels_found": [],
        "category_stats": {},
    }

    all_categories = Counter()
    disease_patient_map = defaultdict(set)
    changed_files = 0

    for disease, patient, stem, json_path, img_path in samples:
        disease_patient_map[disease].add(patient)
        cleaned_data, changes = clean_annotations(json_path, disease)

        # 统计类别
        for obj in cleaned_data.get("objects", []):
            cat = obj.get("category", "")
            all_categories[cat] += 1
            if cat in SUSPICIOUS_LABELS:
                report["suspicious_labels_found"].append({
                    "file": str(json_path),
                    "label": cat,
                })

        if changes:
            changed_files += 1
            report["label_changes"].append({
                "file": str(json_path),
                "changes": changes,
            })

        if not args.dry_run:
            # 输出清洗后的文件
            rel = json_path.relative_to(RAW_DIR)
            out_json = CLEANED_DIR / rel
            out_json.parent.mkdir(parents=True, exist_ok=True)

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

            # 复制/链接图片
            out_img = out_json.parent / img_path.name
            if not out_img.exists():
                shutil.copy2(img_path, out_img)

    # 类别统计
    report["category_stats"] = {
        "total_unique": len(all_categories),
        "categories": dict(all_categories.most_common()),
    }
    report["changed_files"] = changed_files

    # 疾病-患者统计
    report["disease_patient_counts"] = {
        d: len(ps) for d, ps in sorted(disease_patient_map.items())
    }

    # 保存报告
    report_path = CLEAN_REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n清洗完成:")
    print(f"  修改文件数: {changed_files}")
    print(f"  唯一类别数: {len(all_categories)}")
    print(f"  疑似错误标签: {len(report['suspicious_labels_found'])}")
    print(f"  报告已保存: {report_path}")

    if not args.dry_run:
        print(f"  清洗数据已输出到: {CLEANED_DIR}")

        # 将收集到的类别写入 label_mapping 供后续脚本使用
        cats_sorted = sorted(all_categories.keys())
        cat_to_id = {cat: i for i, cat in enumerate(cats_sorted)}
        cats_file = CATEGORY_MAPPING_FILE
        with open(cats_file, "w", encoding="utf-8") as f:
            json.dump({
                "categories": cats_sorted,
                "category_to_id": cat_to_id,
            }, f, ensure_ascii=False, indent=2)
        print(f"  类别映射已保存: {cats_file}")
    else:
        print("  (dry-run 模式，未输出文件)")


if __name__ == "__main__":
    main()
