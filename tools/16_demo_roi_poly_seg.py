"""Gradio demo for ROI polygon segmentation."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (
    CLEANED_DIR,
    ROI_POLY_SEG_ARTIFACTS_DIR,
    SPLITS_DIR,
    TOOLS_DIR,
)
from src.data_utils import load_split_file


def _import_visualize_module():
    script_path = TOOLS_DIR / "15_visualize_roi_poly_seg.py"
    spec = importlib.util.spec_from_file_location("roi_poly_seg_vis", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VIS = _import_visualize_module()
_MODEL = None
_DEVICE = None


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    train_mod = VIS.import_train_module()
    categories, _ = VIS.load_category_mapping(VIS.CATEGORY_MAPPING_FILE)
    model = train_mod.create_model(num_classes=len(categories) + 1)
    ckpt = ROI_POLY_SEG_ARTIFACTS_DIR / "mask_rcnn" / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"模型权重不存在: {ckpt}")
    device = _get_device()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    _MODEL = model
    return _MODEL


def _predict_one(image_path: str, score_thresh: float):
    if not image_path:
        return None, "请先选择图片", []

    img_path = Path(image_path)
    if not img_path.exists():
        return None, f"图片不存在: {img_path}", []

    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        return None, f"缺少同名标注文件: {json_path.name}", []

    categories, category_to_id = VIS.load_category_mapping(VIS.CATEGORY_MAPPING_FILE)
    id_to_name = {idx: name for name, idx in category_to_id.items()}

    model = _load_model()
    device = _get_device()

    roi_image, gt_objects = VIS.load_roi_sample(img_path, json_path)
    train_mod = VIS.import_train_module()

    with torch.no_grad():
        image_tensor = train_mod.F.to_tensor(roi_image).to(device)
        outputs = model([image_tensor])[0]

    pred_polys = VIS.masks_to_polygons(
        outputs["masks"].detach().cpu().numpy()[:, 0],
        outputs["labels"].detach().cpu().numpy(),
        outputs["scores"].detach().cpu().numpy(),
        id_to_name=id_to_name,
        score_thresh=score_thresh,
    )

    pred_overlay = VIS.draw_polygon_overlay(roi_image, pred_polys, (255, 0, 0))
    gt_overlay = VIS.draw_polygon_overlay(roi_image, gt_objects, (0, 255, 0))
    triptych = VIS.make_triptych(roi_image, pred_overlay, gt_overlay)

    table = []
    for item in pred_polys:
        table.append([item["category"], item["score"]])

    info = (
        f"设备: {device}\n"
        f"图片: {img_path.name}\n"
        f"预测区域数: {len(pred_polys)}\n"
        f"人工标注区域数: {len(gt_objects)}"
    )
    return triptych, info, table


def _list_images(folder_path: str) -> List[str]:
    if not folder_path:
        return []
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(folder.glob(ext))
    return [str(p) for p in sorted(images)]


def _refresh_folder(folder_path: str):
    files = _list_images(folder_path)
    if not files:
        return gr.Dropdown(choices=[], value=None), "该目录下没有可用图片"
    return gr.Dropdown(choices=files, value=files[0]), f"找到 {len(files)} 张图片"


def _load_val_examples(limit: int = 20) -> List[str]:
    split_file = SPLITS_DIR / "val.txt"
    if not split_file.exists():
        return []
    rel_paths = load_split_file(split_file)
    return [str(CLEANED_DIR / rel_path) for rel_path in rel_paths[:limit]]


def create_app():
    val_examples = _load_val_examples()

    with gr.Blocks(title="ROI 多边形自动标注演示") as app:
        gr.Markdown(
            "# ROI 多边形自动标注演示\n"
            "输入原始图像，系统会自动按 `ultrasound_rect` 提取超声视野，并输出预测/真值三联图。"
        )

        with gr.Tab("按文件夹选择"):
            with gr.Row():
                folder_input = gr.Textbox(
                    label="图片文件夹",
                    value=str(CLEANED_DIR / "类风湿性关节炎"),
                )
                refresh_btn = gr.Button("读取文件夹")
            folder_status = gr.Textbox(label="状态", interactive=False)
            image_dropdown = gr.Dropdown(label="选择图片", choices=[])

            with gr.Row():
                score_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="分数阈值")
                predict_btn = gr.Button("开始区域预测", variant="primary")

            with gr.Row():
                output_image = gr.Image(label="三联图", type="pil")
                with gr.Column():
                    output_text = gr.Textbox(label="预测信息", lines=6, interactive=False)
                    output_table = gr.Dataframe(headers=["类别", "分数"], label="预测区域")

            refresh_btn.click(
                fn=_refresh_folder,
                inputs=[folder_input],
                outputs=[image_dropdown, folder_status],
            )
            predict_btn.click(
                fn=_predict_one,
                inputs=[image_dropdown, score_slider],
                outputs=[output_image, output_text, output_table],
            )

        with gr.Tab("单张图片"):
            image_path_input = gr.Textbox(label="图片路径")
            score_slider2 = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="分数阈值")
            predict_btn2 = gr.Button("开始区域预测", variant="primary")
            output_image2 = gr.Image(label="三联图", type="pil")
            output_text2 = gr.Textbox(label="预测信息", lines=6, interactive=False)
            output_table2 = gr.Dataframe(headers=["类别", "分数"], label="预测区域")

            predict_btn2.click(
                fn=_predict_one,
                inputs=[image_path_input, score_slider2],
                outputs=[output_image2, output_text2, output_table2],
            )

        if val_examples:
            gr.Examples(
                examples=[[path] for path in val_examples[:10]],
                inputs=[image_path_input],
                label="验证集示例图片",
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="ROI polygon segmentation demo")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
