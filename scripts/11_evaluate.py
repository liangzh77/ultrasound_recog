"""统一评估脚本。

对三个方案的输出统一计算分割和分类指标。

用法:
    python scripts/11_evaluate.py --method yolo
    python scripts/11_evaluate.py --method nnunet
    python scripts/11_evaluate.py --method medsam
    python scripts/11_evaluate.py --compare  # 对比所有方案
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import DISEASE_CLASSES
from src.metrics import compute_classification_metrics, compute_dice, compute_iou
from src.visualize import plot_comparison_bar, plot_confusion_matrix

RESULTS_DIR = ROOT / "runs" / "evaluation"


def evaluate_yolo():
    """评估 YOLO-seg 结果。"""
    from ultralytics import YOLO

    best_weight = ROOT / "runs" / "yolo_seg" / "knee_v1" / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"错误: YOLO 模型权重不存在: {best_weight}")
        return None

    model = YOLO(str(best_weight))
    config_path = ROOT / "configs" / "yolo_seg.yaml"

    # test 集评估
    metrics = model.val(data=str(config_path), split="test")

    results = {
        "method": "YOLO11-seg",
        "segmentation": {
            "mAP50": float(metrics.seg.map50),
            "mAP50-95": float(metrics.seg.map),
        },
        "detection": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
    }
    return results


def evaluate_nnunet():
    """评估 nnU-Net 结果。"""
    pred_dir = ROOT / "runs" / "nnunet" / "predictions"
    gt_dir = ROOT / "data" / "nnunet" / "Dataset001_Knee" / "labelsTs"

    if not pred_dir.exists():
        print(f"错误: nnU-Net 预测结果不存在: {pred_dir}")
        return None

    from PIL import Image

    dice_scores = []
    iou_scores = []

    for pred_file in sorted(pred_dir.glob("*.png")):
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            continue

        pred = np.array(Image.open(pred_file))
        gt = np.array(Image.open(gt_file))

        # 逐类别计算
        for cls_id in range(1, max(pred.max(), gt.max()) + 1):
            p = (pred == cls_id).astype(np.float32)
            t = (gt == cls_id).astype(np.float32)
            if t.sum() == 0 and p.sum() == 0:
                continue
            dice_scores.append(compute_dice(p, t))
            iou_scores.append(compute_iou(p, t))

    results = {
        "method": "nnU-Net",
        "segmentation": {
            "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
            "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            "std_dice": float(np.std(dice_scores)) if dice_scores else 0.0,
        },
    }
    return results


def compare_methods():
    """对比所有方案的结果。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 尝试加载各方案结果
    for method_file in RESULTS_DIR.glob("*_results.json"):
        with open(method_file, encoding="utf-8") as f:
            data = json.load(f)
            all_results[data["method"]] = data

    if not all_results:
        print("没有找到已保存的评估结果。请先运行各方案的评估。")
        return

    print("\n" + "=" * 60)
    print("方案对比结果")
    print("=" * 60)

    for method, data in all_results.items():
        print(f"\n{method}:")
        if "segmentation" in data:
            seg = data["segmentation"]
            for k, v in seg.items():
                print(f"  {k}: {v:.4f}")
        if "classification" in data:
            cls = data["classification"]
            for k, v in cls.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

    # 生成对比图
    if len(all_results) >= 2:
        seg_metrics = {}
        for method, data in all_results.items():
            seg = data.get("segmentation", {})
            seg_metrics[method] = seg

        # 找共同指标
        common_keys = set.intersection(*[set(v.keys()) for v in seg_metrics.values()])
        for key in common_keys:
            if all(isinstance(seg_metrics[m][key], (int, float)) for m in seg_metrics):
                plot_comparison_bar(
                    seg_metrics, key,
                    RESULTS_DIR / f"compare_{key}.png",
                    title=f"方案对比: {key}",
                )
                print(f"\n对比图已保存: {RESULTS_DIR / f'compare_{key}.png'}")


def main():
    parser = argparse.ArgumentParser(description="统一评估")
    parser.add_argument("--method", choices=["yolo", "nnunet", "medsam"], help="评估指定方案")
    parser.add_argument("--compare", action="store_true", help="对比所有方案")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare_methods()
        return

    if args.method == "yolo":
        results = evaluate_yolo()
    elif args.method == "nnunet":
        results = evaluate_nnunet()
    elif args.method == "medsam":
        print("MedSAM 评估请在训练脚本中使用 --evaluate 参数")
        return
    else:
        print("请指定 --method 或 --compare")
        return

    if results:
        save_path = RESULTS_DIR / f"{args.method}_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存: {save_path}")

        for k, v in results.items():
            if isinstance(v, dict):
                print(f"\n{k}:")
                for k2, v2 in v.items():
                    if isinstance(v2, float):
                        print(f"  {k2}: {v2:.4f}")


if __name__ == "__main__":
    main()
