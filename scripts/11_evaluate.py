"""统一评估脚本。

对三个方案的输出统一计算分割和分类指标。

用法:
    python scripts/11_evaluate.py --method yolo
    python scripts/11_evaluate.py --method nnunet
    python scripts/11_evaluate.py --method medsam --sam-weights path/to/medsam_vit_b.pth
    python scripts/11_evaluate.py --method classifier
    python scripts/11_evaluate.py --compare  # 对比所有方案
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import DISEASE_CLASSES, DISEASE_CLASS_TO_ID
from src.metrics import compute_classification_metrics, compute_dice, compute_iou
from src.visualize import plot_comparison_bar, plot_confusion_matrix

RESULTS_DIR = ROOT / "runs" / "evaluation"


def _import_script(name: str):
    """通过文件路径导入以数字开头的脚本模块。"""
    script_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_yolo():
    """评估 YOLO11-seg v2（21 类合并标签）在 test 集上的分割指标。"""
    from ultralytics import YOLO

    best_weight = ROOT / "runs" / "yolo_seg" / "knee_v2" / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"错误: YOLO 模型权重不存在: {best_weight}")
        return None

    model = YOLO(str(best_weight))
    config_path = ROOT / "configs" / "yolo_seg_merged.yaml"

    # test 集评估
    metrics = model.val(data=str(config_path), split="test")

    results = {
        "method": "YOLO11-seg v2",
        "segmentation": {
            "mAP50": float(metrics.seg.map50),
            "mAP50-95": float(metrics.seg.map),
            "box_mAP50": float(metrics.box.map50),
            "box_mAP50-95": float(metrics.box.map),
        },
        "note": "21 类解剖结构分割，无疾病分类（需配合 EfficientNet-B4 分类器）",
    }
    return results


def evaluate_nnunet():
    """评估 nnU-Net 分割结果。

    使用 fold_0 验证集的 PNG 预测结果对比 labelsTr 中的 GT。
    """
    from PIL import Image

    val_pred_dir = (
        ROOT / "runs" / "nnunet" / "Dataset001_Knee"
        / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0" / "validation"
    )
    gt_dir = ROOT / "data" / "nnunet" / "Dataset001_Knee" / "labelsTr"

    if not val_pred_dir.exists():
        print(f"错误: nnU-Net 验证预测不存在: {val_pred_dir}")
        return None

    pred_files = sorted(val_pred_dir.glob("*.png"))
    if not pred_files:
        print("错误: 未找到 nnU-Net 预测 PNG 文件")
        return None

    # 22 类（含 background）
    num_classes = 22
    per_class_dice = {i: [] for i in range(1, num_classes)}
    per_class_iou = {i: [] for i in range(1, num_classes)}

    matched = 0
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            continue

        pred = np.array(Image.open(pred_file))
        gt = np.array(Image.open(gt_file))
        matched += 1

        for cls_id in range(1, num_classes):
            p = (pred == cls_id).astype(np.float32)
            t = (gt == cls_id).astype(np.float32)
            if t.sum() == 0 and p.sum() == 0:
                continue
            per_class_dice[cls_id].append(compute_dice(p, t))
            per_class_iou[cls_id].append(compute_iou(p, t))

    # 加载类别名
    dataset_json = ROOT / "data" / "nnunet" / "Dataset001_Knee" / "dataset.json"
    with open(dataset_json, encoding="utf-8") as f:
        label_map = json.load(f)["labels"]
    id_to_name = {v: k for k, v in label_map.items()}

    # 汇总
    class_results = {}
    all_dice = []
    all_iou = []
    for cls_id in range(1, num_classes):
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        if per_class_dice[cls_id]:
            d = float(np.mean(per_class_dice[cls_id]))
            iou_val = float(np.mean(per_class_iou[cls_id]))
            class_results[name] = {"dice": round(d, 4), "iou": round(iou_val, 4),
                                   "count": len(per_class_dice[cls_id])}
            all_dice.append(d)
            all_iou.append(iou_val)

    results = {
        "method": "nnU-Net",
        "eval_set": f"fold_0 validation ({matched} images)",
        "segmentation": {
            "mean_dice": round(float(np.mean(all_dice)), 4) if all_dice else 0.0,
            "mean_iou": round(float(np.mean(all_iou)), 4) if all_iou else 0.0,
            "std_dice": round(float(np.std(all_dice)), 4) if all_dice else 0.0,
        },
        "per_class": class_results,
    }
    return results


def evaluate_medsam(sam_weights: str | None = None):
    """评估 MedSAM 微调模型在 val 集上的分割 + 分类指标。"""
    import torch
    from torch.utils.data import DataLoader

    model_path = ROOT / "runs" / "medsam" / "best.pth"
    if not model_path.exists():
        print(f"错误: MedSAM 模型权重不存在: {model_path}")
        return None

    # 查找 SAM 预训练权重
    if sam_weights is None:
        candidates = [
            ROOT / "medsam_vit_b.pth",
            ROOT / "weights" / "medsam_vit_b.pth",
            Path.home() / "medsam_vit_b.pth",
        ]
        for c in candidates:
            if c.exists():
                sam_weights = str(c)
                break
        if sam_weights is None:
            print("错误: 未找到 MedSAM 预训练权重 (medsam_vit_b.pth)")
            print("请通过 --sam-weights 指定路径")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 导入训练脚本中的模型类和数据集类
    train_mod = _import_script("10_train_medsam")
    from segment_anything import sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=sam_weights)
    model = train_mod.MedSAMWithClassifier(sam, num_classes=len(DISEASE_CLASSES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 加载验证数据
    EMBED_DIR = ROOT / "data" / "medsam_embeddings"
    use_cached = (EMBED_DIR / "val").exists()

    if use_cached:
        val_ds = train_mod.CachedMedSAMDataset("val")
    else:
        val_ds = train_mod.MedSAMDataset("val")

    val_loader = DataLoader(val_ds, batch_size=8, num_workers=0)

    dice_sum = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if use_cached:
                embeddings, masks, bboxes, disease_ids = batch
                embeddings = embeddings.float().to(device)
            else:
                images, masks, bboxes, disease_ids = batch
                images = images.to(device)
            masks = masks.to(device)
            bboxes = bboxes.to(device)
            disease_ids = disease_ids.to(device)

            if use_cached:
                low_res_masks, _, cls_logits = model.forward_with_embeddings(embeddings, bboxes)
            else:
                low_res_masks, _, cls_logits = model.forward_with_images(images, bboxes)

            # Dice
            pred_masks = (torch.sigmoid(low_res_masks) > 0.5).float()
            intersection = (pred_masks * masks).sum(dim=(2, 3))
            union = pred_masks.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            dice_sum += dice.sum().item()

            # Classification
            _, predicted = cls_logits.max(1)
            correct += predicted.eq(disease_ids).sum().item()
            total += disease_ids.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(disease_ids.cpu().numpy())

    avg_dice = dice_sum / max(total, 1)
    avg_acc = correct / max(total, 1)

    cls_metrics = compute_classification_metrics(all_labels, all_preds, DISEASE_CLASSES)

    results = {
        "method": "MedSAM",
        "eval_set": f"val ({total} samples)",
        "segmentation": {
            "mean_dice": round(avg_dice, 4),
        },
        "classification": {
            "accuracy": round(avg_acc, 4),
            "f1_weighted": round(cls_metrics["f1_weighted"], 4),
            "f1_macro": round(cls_metrics["f1_macro"], 4),
        },
    }

    # 保存混淆矩阵
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(cls_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, DISEASE_CLASSES,
                          RESULTS_DIR / "medsam_confusion_matrix.png",
                          title="MedSAM Classification (Val)")
    print(f"混淆矩阵已保存: {RESULTS_DIR / 'medsam_confusion_matrix.png'}")

    return results


def evaluate_classifier():
    """评估 EfficientNet-B0 分类器在 test 集上的指标。"""
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    ckpt = ROOT / "runs" / "classifier" / "best.pth"
    if not ckpt.exists():
        print(f"错误: 分类器权重不存在: {ckpt}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(DISEASE_CLASSES)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 复用训练脚本中的 Dataset
    classifier_mod = _import_script("09_train_classifier")

    test_ds = classifier_mod.UltrasoundDataset(
        ROOT / "data" / "splits" / "test.txt",
        ROOT / "data" / "cleaned",
        transform,
    )
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=0)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    cls_metrics = compute_classification_metrics(all_labels, all_preds, DISEASE_CLASSES)

    results = {
        "method": "EfficientNet-B0",
        "eval_set": f"test ({len(all_labels)} images)",
        "classification": {
            "accuracy": round(cls_metrics["accuracy"], 4),
            "f1_weighted": round(cls_metrics["f1_weighted"], 4),
            "f1_macro": round(cls_metrics["f1_macro"], 4),
        },
    }

    # 保存混淆矩阵
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(cls_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, DISEASE_CLASSES,
                          RESULTS_DIR / "classifier_confusion_matrix.png",
                          title="EfficientNet-B4 Classification (Test)")
    print(f"混淆矩阵已保存: {RESULTS_DIR / 'classifier_confusion_matrix.png'}")

    return results


def compare_methods():
    """对比所有方案的结果。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
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

    # 分割指标对比
    seg_methods = {m: d["segmentation"] for m, d in all_results.items() if "segmentation" in d}
    if seg_methods:
        print("\n--- 分割指标 ---")
        for method, seg in seg_methods.items():
            print(f"\n  {method}:")
            for k, v in seg.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")

    # 分类指标对比
    cls_methods = {m: d["classification"] for m, d in all_results.items() if "classification" in d}
    if cls_methods:
        print("\n--- 分类指标 ---")
        for method, cls in cls_methods.items():
            print(f"\n  {method}:")
            for k, v in cls.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")

    # 生成对比图 — 分割
    if len(seg_methods) >= 2:
        common_keys = set.intersection(*[set(v.keys()) for v in seg_methods.values()])
        for key in sorted(common_keys):
            if all(isinstance(seg_methods[m][key], (int, float)) for m in seg_methods):
                plot_comparison_bar(
                    seg_methods, key,
                    RESULTS_DIR / f"compare_seg_{key}.png",
                    title=f"分割指标对比: {key}",
                )
                print(f"\n分割对比图: {RESULTS_DIR / f'compare_seg_{key}.png'}")

    # 生成对比图 — 分类
    if len(cls_methods) >= 2:
        common_keys = set.intersection(*[set(v.keys()) for v in cls_methods.values()])
        for key in sorted(common_keys):
            if all(isinstance(cls_methods[m][key], (int, float)) for m in cls_methods):
                plot_comparison_bar(
                    cls_methods, key,
                    RESULTS_DIR / f"compare_cls_{key}.png",
                    title=f"分类指标对比: {key}",
                )
                print(f"\n分类对比图: {RESULTS_DIR / f'compare_cls_{key}.png'}")


def main():
    parser = argparse.ArgumentParser(description="统一评估")
    parser.add_argument("--method", choices=["yolo", "nnunet", "medsam", "classifier"],
                        help="评估指定方案")
    parser.add_argument("--compare", action="store_true", help="对比所有方案")
    parser.add_argument("--sam-weights", type=str, default=None,
                        help="MedSAM 预训练权重路径 (medsam_vit_b.pth)")
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
        results = evaluate_medsam(args.sam_weights)
    elif args.method == "classifier":
        results = evaluate_classifier()
    else:
        print("请指定 --method 或 --compare")
        return

    if results:
        save_path = RESULTS_DIR / f"{args.method}_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存: {save_path}")

        for k, v in results.items():
            if isinstance(v, dict) and k != "per_class":
                print(f"\n{k}:")
                for k2, v2 in v.items():
                    if isinstance(v2, float):
                        print(f"  {k2}: {v2:.4f}")


if __name__ == "__main__":
    main()
