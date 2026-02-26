"""评估指标计算模块。"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """计算 Dice Score。"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """计算 IoU (Jaccard)。"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_per_class_dice(pred_mask: np.ndarray, target_mask: np.ndarray,
                           num_classes: int) -> dict[int, float]:
    """计算逐类别 Dice Score。"""
    results = {}
    for cls_id in range(num_classes):
        p = (pred_mask == cls_id).astype(np.float32)
        t = (target_mask == cls_id).astype(np.float32)
        if t.sum() == 0 and p.sum() == 0:
            results[cls_id] = 1.0
        elif t.sum() == 0:
            results[cls_id] = 0.0
        else:
            results[cls_id] = compute_dice(p, t)
    return results


def compute_classification_metrics(y_true: list, y_pred: list,
                                   class_names: list[str]) -> dict:
    """计算分类指标。"""
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_auc_roc(y_true: list, y_probs: np.ndarray,
                    num_classes: int) -> dict[str, float]:
    """计算 AUC-ROC (one-vs-rest)。"""
    results = {}
    y_true_arr = np.array(y_true)

    for cls_id in range(num_classes):
        binary_true = (y_true_arr == cls_id).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            results[cls_id] = float("nan")
        else:
            results[cls_id] = roc_auc_score(binary_true, y_probs[:, cls_id])

    valid = [v for v in results.values() if not np.isnan(v)]
    results["macro_avg"] = np.mean(valid) if valid else 0.0
    return results
