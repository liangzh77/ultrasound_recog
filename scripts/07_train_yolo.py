"""方案 A: YOLO11-seg 实例分割训练。

训练 YOLO11-seg 模型进行膝关节超声实例分割。
检测到的区域类别前缀用于聚合推断疾病类型。

用法:
    python scripts/07_train_yolo.py [--resume]
    python scripts/07_train_yolo.py --predict <image_path>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import LABEL_PREFIX_TO_DISEASE

CONFIG_PATH = ROOT / "configs" / "yolo_seg.yaml"


def train(resume: bool = False):
    """训练 YOLO11-seg 模型。"""
    from ultralytics import YOLO

    if resume:
        model = YOLO(str(ROOT / "runs" / "yolo_seg" / "knee_v1" / "weights" / "last.pt"))
    else:
        model = YOLO("yolo11m-seg.pt")

    model.train(
        data=str(CONFIG_PATH),
        epochs=300,
        imgsz=640,
        batch=4,
        patience=50,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "yolo_seg"),
        name="knee_v1",
        exist_ok=resume,
        # 类别不平衡
        cls=0.5,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        scale=0.3,
        # 保存
        save=True,
        save_period=50,
        val=True,
    )

    # 在 test 集上评估
    print("\n在 test 集上评估...")
    metrics = model.val(
        data=str(CONFIG_PATH),
        split="test",
    )
    print(f"  mAP50: {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")

    return model


def classify_disease_from_detections(results, class_names: list[str]) -> str:
    """从检测结果的类别前缀聚合推断疾病类型。"""
    disease_votes = Counter()

    for r in results:
        if r.masks is None:
            continue
        for cls_id in r.boxes.cls.cpu().numpy():
            cls_name = class_names[int(cls_id)]
            for prefix, disease in LABEL_PREFIX_TO_DISEASE:
                if cls_name.startswith(prefix):
                    disease_votes[disease] += 1
                    break

    if not disease_votes:
        return "未知"
    return disease_votes.most_common(1)[0][0]


def predict(image_path: str):
    """对单张图片进行预测。"""
    from ultralytics import YOLO

    best_weight = ROOT / "runs" / "yolo_seg" / "knee_v1" / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"错误: 模型权重不存在: {best_weight}")
        sys.exit(1)

    model = YOLO(str(best_weight))
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        project=str(ROOT / "runs" / "yolo_seg" / "predict"),
        conf=0.25,
        iou=0.5,
    )

    # 疾病分类
    class_names = list(model.names.values())
    disease = classify_disease_from_detections(results, class_names)

    print(f"\n预测结果:")
    print(f"  疾病诊断: {disease}")
    if results[0].masks is not None:
        n = len(results[0].masks)
        print(f"  检测到 {n} 个区域")
        for i, cls_id in enumerate(results[0].boxes.cls.cpu().numpy()):
            conf = results[0].boxes.conf[i].item()
            print(f"    [{i+1}] {class_names[int(cls_id)]} ({conf:.2f})")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 训练与推理")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    parser.add_argument("--predict", type=str, default=None, help="对指定图片进行预测")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        train(resume=args.resume)


if __name__ == "__main__":
    main()
