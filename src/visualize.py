"""结果可视化模块。"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          save_path: str | Path, title: str = "Confusion Matrix"):
    """绘制混淆矩阵热力图。"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_bar(metrics_dict: dict[str, dict[str, float]],
                        metric_name: str, save_path: str | Path,
                        title: str = ""):
    """绘制多方案对比柱状图。

    Args:
        metrics_dict: {方案名: {指标名: 值}}
        metric_name: 要对比的指标名
    """
    methods = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_name, 0) for m in methods]
    colors = ["#0071e3", "#34c759", "#af52de", "#ff9500"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, values, color=colors[:len(methods)], width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontsize=11)

    plt.ylabel(metric_name, fontsize=12)
    plt.title(title or f"{metric_name} Comparison", fontsize=14)
    plt.ylim(0, max(values) * 1.15 if values else 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray,
                          alpha: float = 0.4,
                          colors: list[tuple] | None = None) -> np.ndarray:
    """将分割掩码叠加在原图上。

    Args:
        image: RGB 图像 (H, W, 3), uint8
        mask: 标签掩码 (H, W), int
        alpha: 透明度
        colors: 每个类别的 RGB 颜色列表

    Returns:
        叠加后的图像
    """
    if colors is None:
        # 生成随机颜色
        rng = np.random.RandomState(42)
        max_label = mask.max() + 1
        colors = [tuple(rng.randint(50, 255, 3).tolist()) for _ in range(max_label)]

    overlay = image.copy()
    for label_id in range(1, mask.max() + 1):
        if label_id >= len(colors):
            continue
        region = mask == label_id
        color = colors[label_id]
        overlay[region] = (
            (1 - alpha) * image[region] + alpha * np.array(color)
        ).astype(np.uint8)

    return overlay


def save_prediction_visualization(image: np.ndarray, pred_mask: np.ndarray,
                                  gt_mask: np.ndarray | None,
                                  save_path: str | Path,
                                  title: str = ""):
    """保存预测结果可视化（原图 + 预测 + GT 对比）。"""
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    pred_overlay = overlay_mask_on_image(image, pred_mask)
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].axis("off")

    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
