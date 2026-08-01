"""Tkinter desktop UI for ROI polygon segmentation."""

import importlib.util
import sys
import threading
from pathlib import Path

import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import CLEANED_DIR, ROI_POLY_SEG_ARTIFACTS_DIR, TOOLS_DIR


def _import_visualize_module():
    script_path = TOOLS_DIR / "15_visualize_roi_poly_seg.py"
    spec = importlib.util.spec_from_file_location("roi_poly_seg_vis", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VIS = _import_visualize_module()
_MODEL = None
_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    train_mod = VIS.import_train_module()
    categories, _ = VIS.load_category_mapping(VIS.CATEGORY_MAPPING_FILE)
    model = train_mod.create_model(num_classes=len(categories) + 1)
    ckpt = ROI_POLY_SEG_ARTIFACTS_DIR / "mask_rcnn" / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"模型权重不存在: {ckpt}")
    device = get_device()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    _MODEL = model
    return _MODEL


def list_images(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(folder.glob(ext))
    return [str(p) for p in sorted(files)]


def predict_image(image_path: str, score_thresh: float):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")
    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"缺少同名 JSON: {json_path}")

    _, category_to_id = VIS.load_category_mapping(VIS.CATEGORY_MAPPING_FILE)
    id_to_name = {idx: name for name, idx in category_to_id.items()}
    train_mod = VIS.import_train_module()
    model = load_model()
    device = get_device()

    roi_image, gt_objects = VIS.load_roi_sample(img_path, json_path)
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

    lines = [
        f"设备: {device}",
        f"图片: {img_path.name}",
        f"预测区域数: {len(pred_polys)}",
        f"人工标注区域数: {len(gt_objects)}",
        "",
    ]
    for item in pred_polys:
        lines.append(f"{item['category']}  {item['score']:.3f}")
    return triptych, "\n".join(lines)


class RoiPolySegApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI 多边形自动标注")
        self.root.geometry("1500x900")

        self.folder_var = tk.StringVar(value=str(CLEANED_DIR))
        self.image_var = tk.StringVar()
        self.score_var = tk.DoubleVar(value=0.5)
        self.status_var = tk.StringVar(value="就绪")
        self.image_paths = []
        self.tk_image = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="图片文件夹:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var, width=90).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top, text="选择文件夹", command=self.choose_folder).grid(row=0, column=2, padx=5)
        ttk.Button(top, text="读取图片", command=self.refresh_images).grid(row=0, column=3, padx=5)
        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(self.root, padding=10)
        mid.pack(fill=tk.X)

        ttk.Label(mid, text="当前图片:").grid(row=0, column=0, sticky="w")
        self.image_combo = ttk.Combobox(mid, textvariable=self.image_var, width=100, state="readonly")
        self.image_combo.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(mid, text="打开单张图片", command=self.open_image).grid(row=0, column=2, padx=5)
        ttk.Label(mid, text="分数阈值:").grid(row=0, column=3, sticky="e")
        ttk.Spinbox(mid, from_=0.1, to=0.9, increment=0.05, textvariable=self.score_var, width=8).grid(row=0, column=4, padx=5)
        ttk.Button(mid, text="开始区域预测", command=self.run_predict).grid(row=0, column=5, padx=5)
        mid.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=320)
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.image_label = ttk.Label(left)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="预测结果").pack(anchor="w")
        self.result_text = tk.Text(right, wrap="word", width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor="w")

    def choose_folder(self):
        folder = filedialog.askdirectory(initialdir=self.folder_var.get() or str(CLEANED_DIR))
        if folder:
            self.folder_var.set(folder)
            self.refresh_images()

    def refresh_images(self):
        folder = self.folder_var.get().strip()
        self.image_paths = list_images(folder)
        self.image_combo["values"] = self.image_paths
        if self.image_paths:
            self.image_var.set(self.image_paths[0])
            self.status_var.set(f"已读取 {len(self.image_paths)} 张图片")
        else:
            self.image_var.set("")
            self.status_var.set("该目录下没有可用图片")

    def open_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=self.folder_var.get() or str(CLEANED_DIR),
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if file_path:
            self.image_var.set(file_path)
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                self.image_combo["values"] = self.image_paths

    def run_predict(self):
        image_path = self.image_var.get().strip()
        if not image_path:
            messagebox.showwarning("提示", "请先选择图片")
            return

        self.status_var.set("正在预测...")
        self.result_text.delete("1.0", tk.END)

        worker = threading.Thread(
            target=self._predict_worker,
            args=(image_path, float(self.score_var.get())),
            daemon=True,
        )
        worker.start()

    def _predict_worker(self, image_path: str, score_thresh: float):
        try:
            image, text = predict_image(image_path, score_thresh)
            self.root.after(0, lambda: self._update_result(image, text))
        except Exception as exc:
            self.root.after(0, lambda: self._show_error(str(exc)))

    def _update_result(self, image: Image.Image, text: str):
        display = image.copy()
        max_w = 1100
        max_h = 760
        display.thumbnail((max_w, max_h))
        self.tk_image = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.tk_image)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.status_var.set("预测完成")

    def _show_error(self, message: str):
        self.status_var.set("预测失败")
        messagebox.showerror("错误", message)


def main():
    root = tk.Tk()
    app = RoiPolySegApp(root)
    app.refresh_images()
    root.mainloop()


if __name__ == "__main__":
    main()
