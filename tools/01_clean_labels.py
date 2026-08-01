"""标签清洗脚本。

遍历 workspace/data/raw/膝关节已标注/ 下所有 JSON 标注文件：
1. 应用 label_mapping.py 中的修复规则，并移除区域标签中的疾病前缀
2. 输出清洗后的数据到 workspace/data/shared_derived/cleaned/（保持原目录结构）
3. 图片通过软链接/复制关联
4. 生成包含全部原始类别映射的报告 workspace/data/shared_derived/clean_report.json

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
    SUSPICIOUS_LABELS,
    fix_label,
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
    """清洗单个 JSON 文件，返回数据、修改记录和全部类别映射计数。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    changes = []
    category_pairs = Counter()
    for obj in data.get("objects", []):
        old_cat = obj.get("category", "")
        new_cat = fix_label(old_cat, disease)
        category_pairs[(old_cat, new_cat)] += 1

        if new_cat != old_cat:
            changes.append({"old": old_cat, "new": new_cat})
            obj["category"] = new_cat

    return data, changes, category_pairs


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
        "dry_run": args.dry_run,
        "cleaned_output_updated": not args.dry_run,
        "total_samples": len(samples),
        "orphan_jsons": orphan_jsons,
        "missing_json_images": missing_jsons,
        "normalization_policy": (
            "疾病目录表示主要诊断；像素区域标签仅表示疾病无关的解剖结构或病变"
        ),
        "label_changes": [],
        "suspicious_labels_found": [],
        "raw_category_stats": {},
        "category_stats": {},
        "category_mapping": [],
        "all_annotation_raw_category_stats": {},
        "all_annotation_category_stats": {},
        "all_annotation_category_mapping": [],
    }

    raw_categories = Counter()
    all_categories = Counter()
    category_mapping_counts = Counter()
    disease_patient_map = defaultdict(set)
    changed_files = 0

    for disease, patient, stem, json_path, img_path in samples:
        disease_patient_map[disease].add(patient)
        cleaned_data, changes, category_pairs = clean_annotations(json_path, disease)

        for (old_cat, new_cat), count in category_pairs.items():
            raw_categories[old_cat] += count
            category_mapping_counts[(old_cat, new_cat)] += count
            if old_cat in SUSPICIOUS_LABELS:
                report["suspicious_labels_found"].append({
                    "file": str(json_path),
                    "label": old_cat,
                    "normalized": new_cat,
                    "count": count,
                })

        # 统计类别
        for obj in cleaned_data.get("objects", []):
            cat = obj.get("category", "")
            all_categories[cat] += 1

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

    # 类别统计与完整的原始 → 规范化映射
    report["raw_category_stats"] = {
        "total_unique": len(raw_categories),
        "categories": dict(raw_categories.most_common()),
    }
    report["category_stats"] = {
        "total_unique": len(all_categories),
        "categories": dict(all_categories.most_common()),
    }
    report["category_mapping"] = [
        {
            "raw": old_cat,
            "normalized": new_cat,
            "objects": count,
        }
        for (old_cat, new_cat), count in sorted(category_mapping_counts.items())
    ]

    # 孤立 JSON 不进入训练数据，但仍纳入“所有标注类别”审计。
    all_annotation_raw_categories = raw_categories.copy()
    all_annotation_categories = all_categories.copy()
    all_annotation_mapping_counts = category_mapping_counts.copy()
    for orphan_json in orphan_jsons:
        orphan_path = Path(orphan_json)
        disease = orphan_path.relative_to(RAW_DIR).parts[0]
        _, _, category_pairs = clean_annotations(orphan_path, disease)
        for (old_cat, new_cat), count in category_pairs.items():
            all_annotation_raw_categories[old_cat] += count
            all_annotation_categories[new_cat] += count
            all_annotation_mapping_counts[(old_cat, new_cat)] += count

    report["all_annotation_raw_category_stats"] = {
        "total_unique": len(all_annotation_raw_categories),
        "categories": dict(all_annotation_raw_categories.most_common()),
    }
    report["all_annotation_category_stats"] = {
        "total_unique": len(all_annotation_categories),
        "categories": dict(all_annotation_categories.most_common()),
    }
    report["all_annotation_category_mapping"] = [
        {
            "raw": old_cat,
            "normalized": new_cat,
            "objects": count,
        }
        for (old_cat, new_cat), count in sorted(
            all_annotation_mapping_counts.items()
        )
    ]
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
    print(
        "  全部 JSON 原始/规范化类别数: "
        f"{len(all_annotation_raw_categories)} / "
        f"{len(all_annotation_categories)}"
    )
    print(
        "  有效配对原始/规范化类别数: "
        f"{len(raw_categories)} / {len(all_categories)}"
    )
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
