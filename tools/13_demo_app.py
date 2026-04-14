"""膝关节超声 AI 辅助诊断 — 交互式演示工具。

基于 Gradio 构建的 Web UI，集成分割与分类两大功能：
- 分割：YOLO11-seg v2（21 类解剖结构实例分割）
- 分类：EfficientNet-B0（7 类疾病分类）

用法:
    python tools/13_demo_app.py
    python tools/13_demo_app.py --share        # 生成公网链接
    python tools/13_demo_app.py --port 7861     # 指定端口
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import CLEANED_DIR, LEGACY_CLASSIFIER_DIR, LEGACY_YOLO_RUN_DIR, SPLITS_DIR
from src.label_mapping import DISEASE_CLASSES

# ---------------------------------------------------------------------------
# 全局模型缓存（避免每次推理重复加载）
# ---------------------------------------------------------------------------
_yolo_model = None
_cls_model = None
_cls_transform = None
_device = None

# 21 类解剖结构的调色板（BGR → RGB）
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (200, 200, 0), (200, 0, 200),
    (0, 200, 200),
]

# 中文字体（用于在图片上绘制中文标签）
_cn_font = None


def _get_cn_font(size: int = 18):
    """加载中文字体，优先微软雅黑，回退到黑体。"""
    global _cn_font
    if _cn_font is not None and _cn_font.size == size:
        return _cn_font
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(font_path).exists():
            _cn_font = ImageFont.truetype(font_path, size)
            return _cn_font
    _cn_font = ImageFont.load_default()
    return _cn_font


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_yolo():
    """加载 YOLO11-seg v2 模型（单例）。"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = LEGACY_YOLO_RUN_DIR / "knee_v2" / "weights" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO 权重不存在: {model_path}")
        _yolo_model = YOLO(str(model_path))
    return _yolo_model


def _load_classifier():
    """加载 EfficientNet-B0 分类器（单例）。"""
    global _cls_model, _cls_transform
    if _cls_model is None:
        import timm
        import torch
        from torchvision import transforms

        ckpt = LEGACY_CLASSIFIER_DIR / "best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"分类器权重不存在: {ckpt}")

        device = _get_device()
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(DISEASE_CLASSES))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval().to(device)
        _cls_model = model

        _cls_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _cls_model, _cls_transform


# ---------------------------------------------------------------------------
# 核心推理函数
# ---------------------------------------------------------------------------
def run_segmentation(image_np: np.ndarray, conf_threshold: float):
    """YOLO11-seg v2 分割推理。

    Returns:
        overlay: 分割叠加图 (RGB numpy)
        detections: 检测结果列表
    """
    model = _load_yolo()
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        iou=0.5,
        save=False,
        verbose=False,
    )

    r = results[0]
    class_names = list(model.names.values())
    overlay = image_np.copy()
    detections = []

    if r.masks is not None:
        h, w = image_np.shape[:2]
        for i in range(len(r.masks)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            cls_name = class_names[cls_id]
            color = PALETTE[cls_id % len(PALETTE)]

            # 绘制分割掩码
            mask = r.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.5 + np.array(color, dtype=np.float64) * 0.5
            ).astype(np.uint8)

            # 绘制边界框
            bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            detections.append({
                "category": cls_name,
                "confidence": round(conf, 4),
                "bbox": [int(v) for v in bbox],
            })

        # 用 PIL 绘制中文标签（OpenCV putText 不支持中文）
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)
        font = _get_cn_font(size=max(16, h // 50))
        for d in detections:
            bbox = d["bbox"]
            label_text = f"{d['category']} {d['confidence']:.2f}"
            color = PALETTE[class_names.index(d["category"]) % len(PALETTE)]
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            # 标签背景
            draw.rectangle(
                [bbox[0], bbox[1] - th - 6, bbox[0] + tw + 6, bbox[1]],
                fill=color,
            )
            # 标签文字
            draw.text((bbox[0] + 3, bbox[1] - th - 4), label_text,
                      fill=(255, 255, 255), font=font)
        overlay = np.array(pil_img)

    return overlay, detections


def run_classification(image_np: np.ndarray):
    """EfficientNet-B0 疾病分类推理。

    Returns:
        probabilities: dict {disease_name: probability}
    """
    import torch
    from PIL import Image

    model, transform = _load_classifier()
    device = _get_device()

    img_pil = Image.fromarray(image_np)
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {DISEASE_CLASSES[i]: float(probs[i]) for i in range(len(DISEASE_CLASSES))}


# ---------------------------------------------------------------------------
# Gradio 回调
# ---------------------------------------------------------------------------
def on_upload(image_np, conf_threshold):
    """图片上传回调：保存到 State，清空输入框，执行分析。

    上传后立即清空拖拽区，使其始终可接收新图片。
    原图显示在「原图」组件，分割结果显示在「分割结果」组件。
    """
    if image_np is None:
        return None, image_np, image_np, None, {}, ""

    overlay, detections = run_segmentation(image_np, conf_threshold)
    probabilities = run_classification(image_np)
    detail_text = _build_detail(detections, probabilities)

    # 返回: 清空拖拽区, State, 原图, 分割叠加, 概率, 详情
    return None, image_np, image_np, overlay, probabilities, detail_text


def on_rerun(stored_image, conf_threshold):
    """重新分析（调整阈值后手动触发）：使用 State 中缓存的图片。"""
    if stored_image is None:
        return None, {}, "请先上传超声图片"

    overlay, detections = run_segmentation(stored_image, conf_threshold)
    probabilities = run_classification(stored_image)
    detail_text = _build_detail(detections, probabilities)
    return overlay, probabilities, detail_text


def _build_detail(detections, probabilities):
    top_disease = max(probabilities, key=probabilities.get)
    top_prob = probabilities[top_disease]
    lines = [f"## 疾病诊断: {top_disease} ({top_prob:.1%})\n"]
    lines.append(f"**检测到 {len(detections)} 个解剖结构区域**\n")
    if detections:
        lines.append("| 解剖结构 | 置信度 |")
        lines.append("|----------|--------|")
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            lines.append(f"| {d['category']} | {d['confidence']:.2%} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 构建 Gradio UI
# ---------------------------------------------------------------------------
def create_app():
    with gr.Blocks(
        title="膝关节超声 AI 辅助诊断",
        css="""
        /* 隐藏上传区的默认占位提示文字 */
        #upload_box .uploading, #upload_box .upload-text,
        #upload_box span[data-testid="upload-text"],
        #upload_box .wrap .icon-wrap,
        #upload_box .or,
        #upload_box .click-upload {
            display: none !important;
        }
        """,
    ) as app:
        stored_image = gr.State(value=None)

        gr.Markdown(
            "# 膝关节超声 AI 辅助诊断系统\n"
            "分割模型: YOLO11m-seg v2 (21类解剖结构) &nbsp;|&nbsp; "
            "分类模型: EfficientNet-B0 (7类疾病)"
        )

        # ---- 顶栏：上传 + 阈值 + 按钮 紧凑一行 ----
        with gr.Row(equal_height=True):
            input_image = gr.Image(
                label="拖拽或点击上传",
                type="numpy",
                height=80,
                scale=2,
                elem_id="upload_box",
            )
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                label="置信度阈值",
                scale=1,
            )
            run_btn = gr.Button("重新分析", variant="primary", size="lg", scale=1)

        # ---- 主体：原图 | 分割结果 并排 ----
        with gr.Row():
            original_image = gr.Image(label="原图", type="numpy", interactive=False)
            output_image = gr.Image(label="分割结果", type="numpy", interactive=False)

        # ---- 底部：分类概率 | 检测详情 并排 ----
        with gr.Row():
            output_probs = gr.Label(label="疾病分类概率", num_top_classes=7)
            output_detail = gr.Markdown(label="详细结果")

        # 示例图片
        examples = []
        test_list = SPLITS_DIR / "test.txt"
        if test_list.exists():
            with open(test_list, "r", encoding="utf-8") as f:
                test_files = [line.strip() for line in f if line.strip()]
            for rel_path in test_files[:3]:
                img_path = CLEANED_DIR / rel_path
                if img_path.exists():
                    examples.append([str(img_path)])

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_image],
                label="示例图片（来自 test 集）",
            )

        # 核心：上传后自动分析 + 清空拖拽区
        input_image.upload(
            fn=on_upload,
            inputs=[input_image, conf_slider],
            outputs=[input_image, stored_image, original_image, output_image, output_probs, output_detail],
        )
        # 按钮：用缓存图片重新分析（调整阈值后）
        run_btn.click(
            fn=on_rerun,
            inputs=[stored_image, conf_slider],
            outputs=[output_image, output_probs, output_detail],
        )

    return app


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="膝关节超声 AI 辅助诊断演示工具")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网分享链接")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )


if __name__ == "__main__":
    main()
