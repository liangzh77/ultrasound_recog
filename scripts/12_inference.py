"""统一推理脚本。

支持选择 yolo / nnunet / medsam 三种模型后端。
输入一张超声图 → 输出标注区域 + 疾病诊断。

用法:
    python scripts/12_inference.py --method yolo --image path/to/image.jpg
    python scripts/12_inference.py --method nnunet --image path/to/image.jpg
    python scripts/12_inference.py --method yolo --export-onnx  # 导出 ONNX
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import DISEASE_CLASSES, LABEL_PREFIX_TO_DISEASE


def infer_yolo(image_path: str, save_vis: bool = True) -> dict:
    """YOLO-seg 推理。"""
    from ultralytics import YOLO

    model_path = ROOT / "runs" / "yolo_seg" / "knee_v1" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.5,
        save=save_vis,
        project=str(ROOT / "runs" / "inference"),
        name="yolo",
        exist_ok=True,
    )

    r = results[0]
    class_names = list(model.names.values())

    # 收集检测结果
    detections = []
    disease_votes = Counter()

    if r.masks is not None:
        for i in range(len(r.masks)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = class_names[cls_id]
            bbox = r.boxes.xyxy[i].cpu().numpy().tolist()

            # 掩码轮廓
            mask = r.masks.data[i].cpu().numpy()
            contours, _ = cv2.findContours(
                (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = []
            for c in contours:
                if len(c) >= 3:
                    polygons.append(c.squeeze().tolist())

            detections.append({
                "category": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in bbox],
                "polygon_count": len(polygons),
            })

            # 投票
            for prefix, disease in LABEL_PREFIX_TO_DISEASE:
                if cls_name.startswith(prefix):
                    disease_votes[disease] += 1
                    break

    diagnosis = disease_votes.most_common(1)[0][0] if disease_votes else "未知"

    return {
        "method": "YOLO11-seg",
        "image": image_path,
        "diagnosis": diagnosis,
        "diagnosis_confidence": disease_votes,
        "num_detections": len(detections),
        "detections": detections,
    }


def infer_nnunet(image_path: str) -> dict:
    """nnU-Net 推理。"""
    from PIL import Image

    # nnU-Net 需要通过命令行工具推理，这里提供简化版本
    pred_dir = ROOT / "runs" / "nnunet" / "predictions"

    # 加载分类器做疾病诊断
    import torch
    import timm
    from torchvision import transforms

    classifier_path = ROOT / "runs" / "classifier" / "best.pth"
    if not classifier_path.exists():
        return {"method": "nnU-Net", "error": "分类器权重不存在，请先训练"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(DISEASE_CLASSES))
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).item())

    return {
        "method": "nnU-Net + EfficientNet-B4",
        "image": image_path,
        "diagnosis": DISEASE_CLASSES[pred_class],
        "probabilities": {DISEASE_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(DISEASE_CLASSES))},
        "note": "分割结果需通过 nnUNetv2_predict 命令生成",
    }


def export_yolo_onnx():
    """导出 YOLO 模型为 ONNX 格式。"""
    from ultralytics import YOLO

    model_path = ROOT / "runs" / "yolo_seg" / "knee_v1" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    onnx_path = model.export(format="onnx", imgsz=1280, simplify=True)
    print(f"ONNX 模型已导出: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="统一推理")
    parser.add_argument("--method", choices=["yolo", "nnunet", "medsam"], default="yolo")
    parser.add_argument("--image", type=str, help="输入图片路径")
    parser.add_argument("--export-onnx", action="store_true", help="导出 YOLO ONNX 模型")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化结果")
    args = parser.parse_args()

    if args.export_onnx:
        export_yolo_onnx()
        return

    if not args.image:
        print("请指定 --image 参数")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"图片不存在: {args.image}")
        sys.exit(1)

    if args.method == "yolo":
        result = infer_yolo(args.image, save_vis=not args.no_vis)
    elif args.method == "nnunet":
        result = infer_nnunet(args.image)
    elif args.method == "medsam":
        print("MedSAM 推理暂需通过训练脚本实现")
        return

    # 输出结果
    print("\n" + "=" * 50)
    print(f"方法: {result['method']}")
    print(f"疾病诊断: {result.get('diagnosis', 'N/A')}")
    if "num_detections" in result:
        print(f"检测区域数: {result['num_detections']}")
    print("=" * 50)

    # 保存 JSON 结果
    out_dir = ROOT / "runs" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
