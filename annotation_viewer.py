"""
膝关节超声标注查看器
依赖: pip install PySide6
运行: python annotation_viewer.py
"""

import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsPolygonItem, QGraphicsPixmapItem, QGraphicsTextItem,
    QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QToolBar, QStatusBar, QFrame, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPolygonF, QColor, QPen, QBrush, QPixmap, QFont,
    QWheelEvent, QKeySequence, QIcon,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent / "data" / "膝关节已标注"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

DISEASE_BADGE_COLORS = {
    "正常":         ("#e8f5e9", "#2e7d32"),
    "类风湿性关节炎": ("#fce4ec", "#c62828"),
    "骨性关节炎":    ("#fff3e0", "#e65100"),
    "痛风性关节炎":  ("#f3e5f5", "#6a1b9a"),
    "脊柱关节炎":    ("#e3f2fd", "#1565c0"),
    "损伤":         ("#fdf3e3", "#b45309"),
    "滑膜囊肿":      ("#e0f7fa", "#00695c"),
}

POLYGON_ALPHA = 55       # 填充透明度 0-255
POLYGON_BORDER_W = 1.5   # 边框宽度 px
LABEL_FONT_SIZE = 9


# ── 颜色映射（黄金角分布，跨图片固定） ──────────────────────────────────────

def build_color_map(labels: list[str]) -> dict[str, QColor]:
    color_map: dict[str, QColor] = {}
    for i, label in enumerate(sorted(set(labels))):
        hue = int((i * 137.508) % 360)
        color = QColor.fromHsv(hue, 210, 230)
        color_map[label] = color
    return color_map


LABEL_CACHE = Path(__file__).parent / ".label_cache.json"


def collect_all_labels() -> list[str]:
    """扫描数据目录，收集所有 category 名称。结果缓存到 .label_cache.json，避免每次重扫。"""
    # 如果缓存存在且比数据目录新，直接读缓存
    if LABEL_CACHE.exists():
        try:
            cached = json.loads(LABEL_CACHE.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                print(f"从缓存加载 {len(cached)} 个标注类别")
                return cached
        except Exception:
            pass

    print("首次扫描标注类别（约需 15-20 秒，结果将缓存）...")
    labels: set[str] = set()
    for json_path in DATA_ROOT.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for obj in data.get("objects", []):
                cat = obj.get("category", "").strip()
                if cat:
                    labels.add(cat)
        except Exception:
            pass

    result = sorted(labels)
    try:
        LABEL_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"缓存已保存：{LABEL_CACHE}")
    except Exception:
        pass
    return result


# ── 图片查看器（QGraphicsView + 缩放） ───────────────────────────────────────

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(self.renderHints().Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#222222")))
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._annotation_group: list = []   # polygon + text items
        self._show_annotations = True
        self._show_labels = True

    def load_image(self, image_path: Path, color_map: dict[str, QColor]) -> list[str]:
        """加载图片和对应标注，返回本图出现的 label 列表（有序）。"""
        self._scene.clear()
        self._annotation_group.clear()

        # 加载图片
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return []
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)

        # 读取同名 JSON
        json_path = image_path.with_suffix(".json")
        labels_in_image: list[str] = []
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for obj in data.get("objects", []):
                    cat = obj.get("category", "").strip()
                    pts = obj.get("segmentation", [])
                    if not cat or len(pts) < 3:
                        continue
                    color = color_map.get(cat, QColor(200, 200, 200))
                    self._add_polygon(cat, pts, color)
                    if cat not in labels_in_image:
                        labels_in_image.append(cat)
            except Exception:
                pass

        # 适配视口
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._apply_visibility()
        return labels_in_image

    def _add_polygon(self, label: str, pts: list, color: QColor):
        polygon = QPolygonF([QPointF(p[0], p[1]) for p in pts])

        fill = QColor(color)
        fill.setAlpha(POLYGON_ALPHA)
        border = QColor(color)
        border.setAlpha(255)

        poly_item = QGraphicsPolygonItem(polygon)
        poly_item.setBrush(QBrush(fill))
        poly_item.setPen(QPen(border, POLYGON_BORDER_W))
        poly_item.setZValue(1)
        self._scene.addItem(poly_item)
        self._annotation_group.append(poly_item)

        # 重心位置放文字
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        text_item = QGraphicsTextItem(label)
        text_item.setDefaultTextColor(color)
        font = QFont()
        font.setPointSize(LABEL_FONT_SIZE)
        font.setBold(True)
        text_item.setFont(font)
        # 居中偏移
        br = text_item.boundingRect()
        text_item.setPos(cx - br.width() / 2, cy - br.height() / 2)
        text_item.setZValue(2)
        self._scene.addItem(text_item)
        self._annotation_group.append(text_item)

    def _apply_visibility(self):
        for item in self._annotation_group:
            if isinstance(item, QGraphicsPolygonItem):
                item.setVisible(self._show_annotations)
            elif isinstance(item, QGraphicsTextItem):
                item.setVisible(self._show_annotations and self._show_labels)

    def toggle_annotations(self, show: bool):
        self._show_annotations = show
        self._apply_visibility()

    def toggle_labels(self, show: bool):
        self._show_labels = show
        self._apply_visibility()

    def zoom_in(self):
        self.scale(1.25, 1.25)

    def zoom_out(self):
        self.scale(0.8, 0.8)

    def zoom_reset(self):
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
        else:
            self.scale(1 / 1.15, 1 / 1.15)


# ── 文件树 ────────────────────────────────────────────────────────────────────

class FileTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setAnimated(True)
        self.setStyleSheet("""
            QTreeWidget { font-size: 12px; border: none; background: #fafafa; }
            QTreeWidget::item { padding: 3px 4px; }
            QTreeWidget::item:selected { background: #d0e4ff; color: #1a56db; }
            QTreeWidget::item:hover { background: #efefef; }
        """)

    def populate(self, data_root: Path):
        self.clear()
        if not data_root.exists():
            return
        for disease_dir in sorted(data_root.iterdir()):
            if not disease_dir.is_dir():
                continue
            patient_dirs = [d for d in sorted(disease_dir.iterdir()) if d.is_dir()]
            disease_item = QTreeWidgetItem(self, [f"{disease_dir.name}  [{len(patient_dirs)}]"])
            disease_item.setData(0, Qt.ItemDataRole.UserRole, None)
            disease_item.setExpanded(False)
            # 懒加载：先放占位子节点，展开时再填充
            disease_item.setData(0, Qt.ItemDataRole.UserRole + 1, disease_dir)
            placeholder = QTreeWidgetItem(disease_item, [""])
            placeholder.setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")

        self.itemExpanded.connect(self._on_expand)

    def _on_expand(self, item: QTreeWidgetItem):
        # 只处理疾病层（有占位子节点）
        if item.childCount() == 1:
            child = item.child(0)
            if child.data(0, Qt.ItemDataRole.UserRole) == "__placeholder__":
                disease_dir: Path = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if disease_dir is None:
                    return
                item.removeChild(child)
                for patient_dir in sorted(disease_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue
                    # 患者目录名格式通常是 "编号_姓名"，只显示姓名部分
                    display_name = patient_dir.name.split("_", 1)[-1] if "_" in patient_dir.name else patient_dir.name
                    patient_item = QTreeWidgetItem(item, [display_name])
                    patient_item.setData(0, Qt.ItemDataRole.UserRole, None)
                    patient_item.setData(0, Qt.ItemDataRole.UserRole + 1, patient_dir)
                    # 图片子节点（只显示有 JSON 的图片）
                    img_files = sorted(
                        f for f in patient_dir.iterdir()
                        if f.suffix in IMAGE_EXTS and f.with_suffix(".json").exists()
                    )
                    for img in img_files:
                        img_item = QTreeWidgetItem(patient_item, [img.name])
                        img_item.setData(0, Qt.ItemDataRole.UserRole, img)


# ── 右侧信息面板 ──────────────────────────────────────────────────────────────

class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)
        self.setStyleSheet("background: white; font-size: 12px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 图片信息区 ──
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.NoFrame)
        info_frame.setStyleSheet("border-bottom: 1px solid #eee;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 10, 12, 10)
        info_layout.setSpacing(4)

        title = QLabel("图片信息")
        title.setStyleSheet("color: #999; font-size: 10px; text-transform: uppercase; letter-spacing: 1px;")
        info_layout.addWidget(title)

        self._lbl_filename = self._make_info_row(info_layout, "文件名")
        self._lbl_size     = self._make_info_row(info_layout, "尺寸")
        self._lbl_patient  = self._make_info_row(info_layout, "患者")

        disease_label = QLabel("疾病")
        disease_label.setStyleSheet("color: #999; margin-top: 4px;")
        info_layout.addWidget(disease_label)

        self._lbl_disease = QLabel("—")
        self._lbl_disease.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._lbl_disease.setStyleSheet(
            "font-weight: bold; font-size: 12px; padding: 3px 10px;"
            "border-radius: 10px; background: #eee; color: #555;"
            "margin-top: 2px;"
        )
        self._lbl_disease.setFixedHeight(26)
        info_layout.addWidget(self._lbl_disease)

        layout.addWidget(info_frame)

        # ── 区域图例区 ──
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.Shape.NoFrame)
        legend_layout = QVBoxLayout(legend_frame)
        legend_layout.setContentsMargins(12, 10, 12, 10)
        legend_layout.setSpacing(4)

        self._legend_title = QLabel("标注区域 (0)")
        self._legend_title.setStyleSheet("color: #999; font-size: 10px;")
        legend_layout.addWidget(self._legend_title)

        self._legend_list = QListWidget()
        self._legend_list.setStyleSheet("""
            QListWidget { border: none; background: transparent; font-size: 11px; }
            QListWidget::item { padding: 4px 0; border-bottom: 1px solid #f5f5f5; }
        """)
        self._legend_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        legend_layout.addWidget(self._legend_list)

        legend_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(legend_frame, stretch=1)

    def _make_info_row(self, parent_layout: QVBoxLayout, label: str) -> QLabel:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #999; min-width: 44px;")
        val = QLabel("—")
        val.setStyleSheet("color: #222; font-weight: 500;")
        val.setWordWrap(True)
        h.addWidget(lbl)
        h.addWidget(val, stretch=1)
        parent_layout.addWidget(row)
        return val

    def update_info(
        self,
        filename: str,
        size: tuple[int, int] | None,
        patient: str,
        disease: str,
        labels: list[str],
        color_map: dict[str, QColor],
    ):
        self._lbl_filename.setText(filename)
        self._lbl_size.setText(f"{size[0]} × {size[1]}" if size else "—")
        self._lbl_patient.setText(patient)

        # 疾病徽章
        bg, fg = DISEASE_BADGE_COLORS.get(disease, ("#eee", "#555"))
        self._lbl_disease.setText(disease if disease else "—")
        self._lbl_disease.setStyleSheet(
            f"font-weight: bold; font-size: 12px; padding: 3px 10px;"
            f"border-radius: 10px; background: {bg}; color: {fg};"
            f"margin-top: 2px;"
        )

        # 区域图例
        self._legend_list.clear()
        self._legend_title.setText(f"标注区域 ({len(labels)})")
        for label in labels:
            color = color_map.get(label, QColor(200, 200, 200))
            item = QListWidgetItem()
            item.setText(f"  {label}")

            # 颜色图标
            swatch = QPixmap(14, 14)
            swatch.fill(color)
            item.setIcon(QIcon(swatch))
            self._legend_list.addItem(item)

    def clear_info(self):
        self._lbl_filename.setText("—")
        self._lbl_size.setText("—")
        self._lbl_patient.setText("—")
        self._lbl_disease.setText("—")
        self._lbl_disease.setStyleSheet(
            "font-weight: bold; font-size: 12px; padding: 3px 10px;"
            "border-radius: 10px; background: #eee; color: #555; margin-top: 2px;"
        )
        self._legend_list.clear()
        self._legend_title.setText("标注区域 (0)")


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, color_map: dict[str, QColor]):
        super().__init__()
        self.color_map = color_map
        self.setWindowTitle("膝关节超声标注查看器")
        self.resize(1280, 780)

        # ── 工具栏 ──
        toolbar = QToolBar("工具栏", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        self._btn_annotations = QPushButton("标注 ON")
        self._btn_annotations.setCheckable(True)
        self._btn_annotations.setChecked(True)
        self._btn_annotations.setFixedWidth(72)
        self._btn_annotations.clicked.connect(self._toggle_annotations)

        self._btn_labels = QPushButton("名称 ON")
        self._btn_labels.setCheckable(True)
        self._btn_labels.setChecked(True)
        self._btn_labels.setFixedWidth(72)
        self._btn_labels.clicked.connect(self._toggle_labels)

        btn_zoom_in  = QPushButton("＋")
        btn_zoom_out = QPushButton("－")
        btn_zoom_rst = QPushButton("适配")
        btn_crop     = QPushButton("✂ 裁剪")
        btn_crop.setEnabled(False)   # 预留，暂未实现
        btn_crop.setToolTip("裁剪功能（即将推出）")

        for btn in (self._btn_annotations, self._btn_labels,
                    btn_zoom_in, btn_zoom_out, btn_zoom_rst, btn_crop):
            btn.setFixedHeight(28)
            toolbar.addWidget(btn)

        btn_zoom_in.clicked.connect(lambda: self._viewer.zoom_in())
        btn_zoom_out.clicked.connect(lambda: self._viewer.zoom_out())
        btn_zoom_rst.clicked.connect(lambda: self._viewer.zoom_reset())

        self._update_toolbar_style()

        # ── 主体三栏 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # 左：文件树
        self._tree = FileTree()
        self._tree.setMinimumWidth(180)
        self._tree.setMaximumWidth(300)
        self._tree.populate(DATA_ROOT)
        self._tree.itemClicked.connect(self._on_tree_click)
        splitter.addWidget(self._tree)

        # 中：图片查看器
        self._viewer = ImageViewer()
        splitter.addWidget(self._viewer)

        # 右：信息面板
        self._panel = InfoPanel()
        splitter.addWidget(self._panel)

        splitter.setSizes([220, 820, 240])
        self.setCentralWidget(splitter)

        # ── 状态栏 ──
        self._status = QStatusBar()
        self._status.setStyleSheet("font-size: 11px; color: #666;")
        self.setStatusBar(self._status)
        self._status.showMessage("请从左侧文件树选择一张图片")

    # ── 事件 ──

    def _on_tree_click(self, item: QTreeWidgetItem, _col: int):
        img_path: Path | None = item.data(0, Qt.ItemDataRole.UserRole)
        if img_path is None or not isinstance(img_path, Path):
            return

        disease = img_path.parent.parent.name   # 疾病目录名
        patient_raw = img_path.parent.name       # 编号_姓名
        patient = patient_raw.split("_", 1)[-1] if "_" in patient_raw else patient_raw

        # 加载图片尺寸
        pixmap = QPixmap(str(img_path))
        size = (pixmap.width(), pixmap.height()) if not pixmap.isNull() else None

        labels = self._viewer.load_image(img_path, self.color_map)
        self._panel.update_info(img_path.name, size, patient, disease, labels, self.color_map)
        self._status.showMessage(
            f"{disease} / {patient} / {img_path.name}   |   {len(labels)} 个标注区域"
        )

    def _toggle_annotations(self):
        show = self._btn_annotations.isChecked()
        self._btn_annotations.setText("标注 ON" if show else "标注 OFF")
        self._viewer.toggle_annotations(show)
        self._update_toolbar_style()

    def _toggle_labels(self):
        show = self._btn_labels.isChecked()
        self._btn_labels.setText("名称 ON" if show else "名称 OFF")
        self._viewer.toggle_labels(show)
        self._update_toolbar_style()

    def _update_toolbar_style(self):
        active_style = (
            "QPushButton { background: #3a6fd8; color: white; border: 1px solid #3a6fd8;"
            "border-radius: 4px; padding: 2px 8px; font-size: 12px; }"
        )
        inactive_style = (
            "QPushButton { background: transparent; color: #555; border: 1px solid #ccc;"
            "border-radius: 4px; padding: 2px 8px; font-size: 12px; }"
            "QPushButton:hover { background: #f0f0f0; }"
        )
        self._btn_annotations.setStyleSheet(
            active_style if self._btn_annotations.isChecked() else inactive_style
        )
        self._btn_labels.setStyleSheet(
            active_style if self._btn_labels.isChecked() else inactive_style
        )


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 启动时扫描所有标注类别，建立颜色映射
    print("正在扫描标注类别...")
    all_labels = collect_all_labels()
    color_map = build_color_map(all_labels)
    print(f"发现 {len(all_labels)} 个标注类别，颜色映射已建立")

    window = MainWindow(color_map)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
