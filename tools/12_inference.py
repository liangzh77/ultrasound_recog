"""统一推理脚本。

支持选择 yolo / nnunet / medsam 三种模型后端。
输入一张超声图 → 输出标注区域 + 疾病诊断。

用法:
    python tools/12_inference.py --method yolo --image path/to/image.jpg
    python tools/12_inference.py --method nnunet --image path/to/image.jpg
    python tools/12_inference.py --method medsam --image path/to/image.jpg --sam-weights medsam_vit_b.pth
    python tools/12_inference.py --method yolo --export-onnx  # 导出 ONNX
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (
    LEGACY_CLASSIFIER_DIR,
    LEGACY_INFERENCE_DIR,
    LEGACY_MEDSAM_RUN_DIR,
    LEGACY_YOLO_RUN_DIR,
    TOOLS_DIR,
)
from src.label_mapping import DISEASE_CLASSES, DISEASE_CLASS_TO_ID


def _import_script(name: str):
    """通过文件路径导入以数字开头的脚本模块。"""
    script_path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classify_image(image_path: str) -> dict:
    """使用 EfficientNet-B0 对图片做疾病分类。"""
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    ckpt = LEGACY_CLASSIFIER_DIR / "best.pth"
    if not ckpt.exists():
        return {"diagnosis": "未知", "error": "分类器权重不存在"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(DISEASE_CLASSES))
    model.load_state_dict(torch.load(ckpt, map_location=device))
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
        "diagnosis": DISEASE_CLASSES[pred_class],
        "probabilities": {DISEASE_CLASSES[i]: round(float(probs[i]), 4)
                          for i in range(len(DISEASE_CLASSES))},
    }


def infer_yolo(image_path: str, save_vis: bool = True) -> dict:
    """YOLO11-seg v2 推理（21 类解剖结构分割 + EfficientNet-B0 疾病分类）。"""
    from ultralytics import YOLO

    model_path = LEGACY_YOLO_RUN_DIR / "knee_v2" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.5,
        save=save_vis,
        project=str(LEGACY_INFERENCE_DIR),
        name="yolo",
        exist_ok=True,
    )

    r = results[0]
    class_names = list(model.names.values())

    # 收集检测结果（21 类解剖结构）
    detections = []
    if r.masks is not None:
        for i in range(len(r.masks)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = class_names[cls_id]
            bbox = r.boxes.xyxy[i].cpu().numpy().tolist()

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

    # 使用 EfficientNet-B0 做疾病诊断
    cls_result = _classify_image(image_path)

    return {
        "method": "YOLO11-seg v2 + EfficientNet-B0",
        "image": image_path,
        "diagnosis": cls_result.get("diagnosis", "未知"),
        "probabilities": cls_result.get("probabilities"),
        "num_detections": len(detections),
        "detections": detections,
    }


def infer_nnunet(image_path: str) -> dict:
    """nnU-Net + EfficientNet-B0 推理。"""
    # 使用 EfficientNet-B0 做疾病诊断
    cls_result = _classify_image(image_path)

    return {
        "method": "nnU-Net + EfficientNet-B0",
        "image": image_path,
        "diagnosis": cls_result.get("diagnosis", "未知"),
        "probabilities": cls_result.get("probabilities"),
        "note": "分割结果需通过 nnUNetv2_predict 命令生成",
    }


def infer_medsam(image_path: str, sam_weights: str) -> dict:
    """MedSAM 推理（分割 + 分类）。

    需要提供 bbox prompt。这里使用整张图作为 bbox（全图分割模式）。
    """
    import torch
    from PIL import Image

    model_path = LEGACY_MEDSAM_RUN_DIR / "best.pth"
    if not model_path.exists():
        return {"method": "MedSAM", "error": "模型权重不存在"}

    if not sam_weights or not Path(sam_weights).exists():
        return {"method": "MedSAM", "error": f"SAM 预训练权重不存在: {sam_weights}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 导入模型类
    train_mod = _import_script("10_train_medsam")
    from segment_anything import sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=sam_weights)
    model = train_mod.MedSAMWithClassifier(sam, num_classes=len(DISEASE_CLASSES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 加载图片
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = np.array(img.resize((1024, 1024), Image.BILINEAR), dtype=np.float32)
    img_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 使用整张图作为 bbox prompt
    bbox = torch.tensor([0.0, 0.0, 1024.0, 1024.0], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        low_res_masks, _, cls_logits = model.forward_with_images(img_tensor, bbox)

        pred_mask = (torch.sigmoid(low_res_masks[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_class = int(cls_logits.argmax(dim=1).item())

    # 保存可视化
    out_dir = LEGACY_INFERENCE_DIR / "medsam"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    overlay = np.array(img)
    overlay[mask_resized > 0] = (
        overlay[mask_resized > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{Path(image_path).stem}_mask.png"), mask_resized * 255)
    cv2.imwrite(str(out_dir / f"{Path(image_path).stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "method": "MedSAM",
        "image": image_path,
        "diagnosis": DISEASE_CLASSES[pred_class],
        "probabilities": {DISEASE_CLASSES[i]: round(float(probs[i]), 4)
                          for i in range(len(DISEASE_CLASSES))},
        "mask_saved": str(out_dir / f"{Path(image_path).stem}_mask.png"),
    }


def export_yolo_onnx():
    """导出 YOLO v2 模型为 ONNX 格式。"""
    from ultralytics import YOLO

    model_path = LEGACY_YOLO_RUN_DIR / "knee_v2" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"ONNX 模型已导出: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="统一推理")
    parser.add_argument("--method", choices=["yolo", "nnunet", "medsam"], default="yolo")
    parser.add_argument("--image", type=str, help="输入图片路径")
    parser.add_argument("--export-onnx", action="store_true", help="导出 YOLO ONNX 模型")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化结果")
    parser.add_argument("--sam-weights", type=str, default=None,
                        help="MedSAM 预训练权重路径 (medsam_vit_b.pth)")
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
        result = infer_medsam(args.image, args.sam_weights)

    # 输出结果
    print("\n" + "=" * 50)
    print(f"方法: {result['method']}")
    print(f"疾病诊断: {result.get('diagnosis', 'N/A')}")
    if "num_detections" in result:
        print(f"检测区域数: {result['num_detections']}")
    if "probabilities" in result and result["probabilities"]:
        print("各类概率:")
        for cls, prob in sorted(result["probabilities"].items(),
                                key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.4f}")
    print("=" * 50)

    # 保存 JSON 结果
    out_dir = LEGACY_INFERENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
