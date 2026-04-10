"""
膝关节超声标注查看器（含裁剪功能）
依赖: pip install PySide6
运行: python annotation_viewer.py
"""

import sys
import json
import copy
from enum import Enum
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsPolygonItem, QGraphicsPixmapItem, QGraphicsTextItem,
    QGraphicsRectItem, QGraphicsPathItem,
    QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QToolBar, QStatusBar, QFrame, QPushButton, QSizePolicy,
    QMessageBox, QFileDialog,
)
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal
from PySide6.QtGui import (
    QPolygonF, QColor, QPen, QBrush, QPixmap, QFont,
    QWheelEvent, QIcon, QPainterPath, QCursor, QShortcut, QKeySequence,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────

DATA_ROOT   = Path(__file__).parent / "data" / "膝关节已标注"
CROP_ROOT   = Path(__file__).parent / "data" / "裁剪结果"
LABEL_CACHE = Path(__file__).parent / ".label_cache.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

DISEASE_BADGE_COLORS = {
    "正常":          ("#e8f5e9", "#2e7d32"),
    "类风湿性关节炎": ("#fce4ec", "#c62828"),
    "骨性关节炎":    ("#fff3e0", "#e65100"),
    "痛风性关节炎":  ("#f3e5f5", "#6a1b9a"),
    "脊柱关节炎":    ("#e3f2fd", "#1565c0"),
    "损伤":          ("#fdf3e3", "#b45309"),
    "滑膜囊肿":      ("#e0f7fa", "#00695c"),
}

POLYGON_ALPHA   = 55
POLYGON_BORDER_W = 1.5
LABEL_FONT_SIZE = 9

HANDLE_SIZE = 10   # 控制点正方形边长（像素，视口坐标）
HANDLE_NAMES = ["TL", "T", "TR", "R", "BR", "B", "BL", "L"]


# ── 颜色映射 ──────────────────────────────────────────────────────────────────

def build_color_map(labels: list[str]) -> dict[str, QColor]:
    color_map: dict[str, QColor] = {}
    for i, label in enumerate(sorted(set(labels))):
        hue = int((i * 137.508) % 360)
        color_map[label] = QColor.fromHsv(hue, 210, 230)
    return color_map


def collect_all_labels() -> list[str]:
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
    for jp in DATA_ROOT.rglob("*.json"):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            for obj in d.get("objects", []):
                c = obj.get("category", "").strip()
                if c:
                    labels.add(c)
        except Exception:
            pass
    result = sorted(labels)
    try:
        LABEL_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return result


# ── 多边形裁剪（Sutherland-Hodgman） ──────────────────────────────────────────

def _intersect(p1, p2, edge_x1, edge_y1, edge_x2, edge_y2):
    """计算线段 p1-p2 与裁剪边的交点。"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ex, ey = edge_x2 - edge_x1, edge_y2 - edge_y1
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-10:
        return p1
    t = ((edge_x1 - p1[0]) * ey - (edge_y1 - p1[1]) * ex) / denom
    return [p1[0] + t * dx, p1[1] + t * dy]


def _inside(p, ex1, ey1, ex2, ey2):
    """判断点是否在裁剪边的内侧（左侧）。"""
    return (ex2 - ex1) * (p[1] - ey1) - (ey2 - ey1) * (p[0] - ex1) >= 0


def clip_polygon_to_rect(pts: list, x1: float, y1: float, x2: float, y2: float) -> list:
    """将多边形裁剪到矩形 (x1,y1)-(x2,y2)，返回裁剪后的点列表。"""
    # 四条裁剪边（屏幕坐标系，内侧为左）
    edges = [
        (x1, y2, x1, y1),   # 左边，向上
        (x1, y1, x2, y1),   # 上边，向右
        (x2, y1, x2, y2),   # 右边，向下
        (x2, y2, x1, y2),   # 下边，向左
    ]
    output = [list(p) for p in pts]
    for ex1, ey1, ex2, ey2 in edges:
        if not output:
            return []
        inp = output
        output = []
        for i, cur in enumerate(inp):
            prev = inp[i - 1]
            if _inside(cur, ex1, ey1, ex2, ey2):
                if not _inside(prev, ex1, ey1, ex2, ey2):
                    output.append(_intersect(prev, cur, ex1, ey1, ex2, ey2))
                output.append(cur)
            elif _inside(prev, ex1, ey1, ex2, ey2):
                output.append(_intersect(prev, cur, ex1, ey1, ex2, ey2))
    return output


def save_crop(image_path: Path, crop_rect: QRectF) -> dict:
    """
    裁剪图片并更新标注 JSON，直接覆盖原文件。
    返回 undo/redo 缓存字典：
      {path, orig_img, orig_json, new_img, new_json}
    其中 orig_* 是覆盖前的内容，new_* 是覆盖后的内容。
    """
    x1 = int(round(crop_rect.x()))
    y1 = int(round(crop_rect.y()))
    x2 = int(round(crop_rect.x() + crop_rect.width()))
    y2 = int(round(crop_rect.y() + crop_rect.height()))
    cw, ch = x2 - x1, y2 - y1

    json_path = image_path.with_suffix(".json")

    # ── 先读原始内容（用于 undo）──
    orig_img_bytes = image_path.read_bytes()
    orig_json_str  = json_path.read_text(encoding="utf-8") if json_path.exists() else None

    # ── 裁剪并覆盖图片 ──
    pixmap  = QPixmap(str(image_path))
    cropped = pixmap.copy(x1, y1, cw, ch)
    cropped.save(str(image_path))

    # ── 更新标注 JSON（覆盖原 JSON）──
    new_json_str = None
    if orig_json_str is not None:
        data     = json.loads(orig_json_str)
        new_data = copy.deepcopy(data)
        new_data["info"]["name"]   = image_path.name
        new_data["info"]["width"]  = cw
        new_data["info"]["height"] = ch
        new_objects = []
        for obj in new_data.get("objects", []):
            pts = obj.get("segmentation", [])
            if len(pts) < 3:
                continue
            clipped = clip_polygon_to_rect(pts, x1, y1, x2, y2)
            if len(clipped) < 3:
                continue
            obj["segmentation"] = [[p[0] - x1, p[1] - y1] for p in clipped]
            new_objects.append(obj)
        new_data["objects"] = new_objects
        new_json_str = json.dumps(new_data, ensure_ascii=False, indent=4)
        json_path.write_text(new_json_str, encoding="utf-8")

    return {
        "path":      image_path,
        "orig_img":  orig_img_bytes,
        "orig_json": orig_json_str,
        "new_img":   image_path.read_bytes(),   # 保存后再读，得到实际写入内容
        "new_json":  new_json_str,
    }


# ── 裁剪覆盖层（控制点 + 选框） ───────────────────────────────────────────────

class CropOverlay:
    """管理场景中的裁剪矩形、遮罩、8 个控制点。"""

    CURSORS = {
        "TL": Qt.CursorShape.SizeFDiagCursor,
        "BR": Qt.CursorShape.SizeFDiagCursor,
        "TR": Qt.CursorShape.SizeBDiagCursor,
        "BL": Qt.CursorShape.SizeBDiagCursor,
        "T":  Qt.CursorShape.SizeVerCursor,
        "B":  Qt.CursorShape.SizeVerCursor,
        "L":  Qt.CursorShape.SizeHorCursor,
        "R":  Qt.CursorShape.SizeHorCursor,
    }

    def __init__(self, scene: QGraphicsScene):
        self._scene = scene
        self._overlay: QGraphicsPathItem | None = None
        self._border: QGraphicsRectItem | None = None
        self._handles: dict[str, QGraphicsRectItem] = {}
        self._img_rect = QRectF()
        self._crop_rect = QRectF()

    def activate(self, img_rect: QRectF, initial: QRectF | None = None):
        self.clear()
        self._img_rect = img_rect
        self._crop_rect = initial if initial else QRectF()

        # 遮罩（裁剪框外半透明黑色）
        self._overlay = QGraphicsPathItem()
        self._overlay.setBrush(QBrush(QColor(0, 0, 0, 110)))
        self._overlay.setPen(QPen(Qt.PenStyle.NoPen))
        self._overlay.setZValue(10)
        self._scene.addItem(self._overlay)

        # 裁剪框边框（白色虚线）
        pen = QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine)
        self._border = QGraphicsRectItem()
        self._border.setPen(pen)
        self._border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._border.setZValue(11)
        self._scene.addItem(self._border)

        # 8 个控制点
        for name in HANDLE_NAMES:
            h = QGraphicsRectItem()
            h.setPen(QPen(QColor(255, 255, 255), 1))
            h.setBrush(QBrush(QColor(30, 120, 255)))
            h.setZValue(12)
            self._scene.addItem(h)
            self._handles[name] = h

        if initial:
            self._update_items()

    def clear(self):
        for item in [self._overlay, self._border] + list(self._handles.values()):
            if item and item.scene():
                self._scene.removeItem(item)
        self._overlay = None
        self._border = None
        self._handles.clear()
        self._crop_rect = QRectF()

    def set_rect(self, rect: QRectF):
        self._crop_rect = rect.normalized()
        self._update_items()

    def get_rect(self) -> QRectF:
        return self._crop_rect.normalized()

    def is_valid(self) -> bool:
        r = self._crop_rect.normalized()
        return r.width() > 4 and r.height() > 4

    def _update_items(self):
        if not self._border:
            return
        r = self._crop_rect.normalized()
        self._border.setRect(r)

        # 遮罩路径（整图减去裁剪框）
        path = QPainterPath()
        path.addRect(self._img_rect)
        path.addRect(r)
        path.setFillRule(Qt.FillRule.OddEvenFill)
        self._overlay.setPath(path)

        # 控制点位置（场景坐标）
        cx = r.x() + r.width() / 2
        cy = r.y() + r.height() / 2
        positions = {
            "TL": (r.x(),       r.y()),
            "T":  (cx,          r.y()),
            "TR": (r.right(),   r.y()),
            "R":  (r.right(),   cy),
            "BR": (r.right(),   r.bottom()),
            "B":  (cx,          r.bottom()),
            "BL": (r.x(),       r.bottom()),
            "L":  (r.x(),       cy),
        }
        # 控制点大小随场景缩放调整（固定视口像素大小）
        hs = HANDLE_SIZE
        for name, (hx, hy) in positions.items():
            self._handles[name].setRect(hx - hs / 2, hy - hs / 2, hs, hs)

    def handle_at(self, scene_pos: QPointF, tol_scene: float) -> str | None:
        """返回鼠标下的控制点名称，tol_scene 为容差（场景坐标）。"""
        for name, h in self._handles.items():
            if h.rect().adjusted(-tol_scene, -tol_scene, tol_scene, tol_scene).contains(scene_pos):
                return name
        return None

    def apply_handle_drag(self, name: str, delta: QPointF):
        """根据控制点拖动方向更新裁剪矩形。"""
        r = self._crop_rect
        dx, dy = delta.x(), delta.y()
        if "L" in name:
            r.setX(r.x() + dx)
        if "R" in name:
            r.setWidth(r.width() + dx)
        if "T" in name:
            r.setY(r.y() + dy)
        if "B" in name:
            r.setHeight(r.height() + dy)
        self._crop_rect = r
        self._update_items()

    def move_rect(self, delta: QPointF):
        self._crop_rect.translate(delta)
        self._update_items()


# ── 图片查看器（含裁剪模式） ──────────────────────────────────────────────────

class CropMode(Enum):
    OFF      = 0
    DRAWING  = 1   # 正在拖拽画矩形
    ADJUSTING = 2  # 可拖动控制点或移动矩形


class ImageViewer(QGraphicsView):

    crop_rect_changed = Signal(QRectF)   # 裁剪框变化时通知主窗口

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
        self._image_path: Path | None = None
        self._img_size: tuple[int, int] = (0, 0)
        self._annotation_group: list = []
        self._show_annotations = True
        self._show_labels = True

        self._crop_mode = CropMode.OFF
        self._crop_overlay = CropOverlay(self._scene)
        self._drag_start: QPointF | None = None
        self._drag_handle: str | None = None
        self._drag_move: bool = False
        self._last_scene_pos: QPointF | None = None

    # ── 图片加载 ──

    def load_image(self, image_path: Path, color_map: dict[str, QColor]) -> list[str]:
        # 退出裁剪模式
        if self._crop_mode != CropMode.OFF:
            self._exit_crop_mode()

        self._scene.clear()
        self._annotation_group.clear()
        self._image_path = image_path

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return []
        self._img_size = (pixmap.width(), pixmap.height())
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)

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

        self._scene.setSceneRect(QRectF(0, 0, *self._img_size))
        self.fitInView(QRectF(0, 0, *self._img_size), Qt.AspectRatioMode.KeepAspectRatio)
        self._apply_visibility()
        return labels_in_image

    def _add_polygon(self, label: str, pts: list, color: QColor):
        polygon = QPolygonF([QPointF(p[0], p[1]) for p in pts])
        fill = QColor(color); fill.setAlpha(POLYGON_ALPHA)
        border = QColor(color); border.setAlpha(255)

        poly_item = QGraphicsPolygonItem(polygon)
        poly_item.setBrush(QBrush(fill))
        poly_item.setPen(QPen(border, POLYGON_BORDER_W))
        poly_item.setZValue(1)
        self._scene.addItem(poly_item)
        self._annotation_group.append(poly_item)

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        text_item = QGraphicsTextItem(label)
        text_item.setDefaultTextColor(color)
        font = QFont(); font.setPointSize(LABEL_FONT_SIZE); font.setBold(True)
        text_item.setFont(font)
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

    # ── 缩放 ──

    def zoom_in(self):  self.scale(1.25, 1.25)
    def zoom_out(self): self.scale(0.8, 0.8)

    def zoom_reset(self):
        if self._img_size[0] > 0:
            self.fitInView(QRectF(0, 0, *self._img_size), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if self._crop_mode == CropMode.OFF:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)

    # ── 裁剪模式 ──

    def enter_crop_mode(self):
        if self._pixmap_item is None:
            return
        self._crop_mode = CropMode.DRAWING
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        img_rect = QRectF(0, 0, *self._img_size)
        self._crop_overlay.activate(img_rect)

    def _exit_crop_mode(self):
        self._crop_mode = CropMode.OFF
        self._drag_start = None
        self._drag_handle = None
        self._drag_move = False
        self._crop_overlay.clear()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def cancel_crop(self):
        self._exit_crop_mode()

    def get_crop_rect(self) -> QRectF | None:
        if self._crop_mode == CropMode.ADJUSTING and self._crop_overlay.is_valid():
            return self._crop_overlay.get_rect()
        return None

    def get_image_path(self) -> Path | None:
        return self._image_path

    # ── 控制点容差（从视口像素转场景坐标） ──

    def _handle_tol(self) -> float:
        inv, _ = self.transform().inverted()
        return inv.m11() * (HANDLE_SIZE * 0.8)

    # ── 鼠标事件 ──

    def mousePressEvent(self, event):
        if self._crop_mode == CropMode.OFF:
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        scene_pos = self.mapToScene(event.pos())

        if self._crop_mode == CropMode.DRAWING:
            self._drag_start = scene_pos
            self._crop_overlay.set_rect(QRectF(scene_pos, scene_pos))

        elif self._crop_mode == CropMode.ADJUSTING:
            handle = self._crop_overlay.handle_at(scene_pos, self._handle_tol())
            if handle:
                self._drag_handle = handle
                self.setCursor(CropOverlay.CURSORS[handle])
            elif self._crop_overlay.get_rect().contains(scene_pos):
                self._drag_move = True
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                # 在框外点击 → 重新画
                self._crop_mode = CropMode.DRAWING
                self._drag_start = scene_pos
                self._crop_overlay.set_rect(QRectF(scene_pos, scene_pos))

        self._last_scene_pos = scene_pos

    def mouseMoveEvent(self, event):
        if self._crop_mode == CropMode.OFF:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if self._crop_mode == CropMode.DRAWING and self._drag_start:
            self._crop_overlay.set_rect(QRectF(self._drag_start, scene_pos))
            self.crop_rect_changed.emit(self._crop_overlay.get_rect())

        elif self._crop_mode == CropMode.ADJUSTING:
            if self._last_scene_pos is None:
                self._last_scene_pos = scene_pos
                return
            delta = scene_pos - self._last_scene_pos

            if self._drag_handle:
                self._crop_overlay.apply_handle_drag(self._drag_handle, delta)
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
            elif self._drag_move:
                self._crop_overlay.move_rect(delta)
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
            else:
                # 悬停时更新光标
                handle = self._crop_overlay.handle_at(scene_pos, self._handle_tol())
                if handle:
                    self.setCursor(CropOverlay.CURSORS[handle])
                elif self._crop_overlay.get_rect().contains(scene_pos):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.setCursor(Qt.CursorShape.CrossCursor)

        self._last_scene_pos = scene_pos

    def mouseReleaseEvent(self, event):
        if self._crop_mode == CropMode.OFF:
            super().mouseReleaseEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._crop_mode == CropMode.DRAWING:
            self._crop_mode = CropMode.ADJUSTING
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self._drag_handle = None
        self._drag_move = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._crop_mode != CropMode.OFF:
            self._exit_crop_mode()
            self.crop_rect_changed.emit(QRectF())
        else:
            super().keyPressEvent(event)


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
            disease_item.setData(0, Qt.ItemDataRole.UserRole + 1, disease_dir)
            QTreeWidgetItem(disease_item, [""]).setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")
        self.itemExpanded.connect(self._on_expand)

    def _on_expand(self, item: QTreeWidgetItem):
        if item.childCount() == 1:
            child = item.child(0)
            if child.data(0, Qt.ItemDataRole.UserRole) != "__placeholder__":
                return
            disease_dir: Path = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if not disease_dir:
                return
            item.removeChild(child)
            for patient_dir in sorted(disease_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                display = patient_dir.name.split("_", 1)[-1] if "_" in patient_dir.name else patient_dir.name
                p_item = QTreeWidgetItem(item, [display])
                p_item.setData(0, Qt.ItemDataRole.UserRole, None)
                p_item.setData(0, Qt.ItemDataRole.UserRole + 1, patient_dir)
                for img in sorted(f for f in patient_dir.iterdir()
                                  if f.suffix in IMAGE_EXTS and f.with_suffix(".json").exists()):
                    img_item = QTreeWidgetItem(p_item, [img.name])
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

        # 图片信息
        info_frame = QFrame()
        info_frame.setStyleSheet("border-bottom: 1px solid #eee;")
        il = QVBoxLayout(info_frame)
        il.setContentsMargins(12, 10, 12, 10)
        il.setSpacing(4)
        QLabel("图片信息", styleSheet="color:#999;font-size:10px;").setParent(info_frame)
        il.addWidget(QLabel("图片信息", styleSheet="color:#999;font-size:10px;"))
        self._lbl_filename = self._row(il, "文件名")
        self._lbl_size     = self._row(il, "尺寸")
        self._lbl_patient  = self._row(il, "患者")
        il.addWidget(QLabel("疾病", styleSheet="color:#999;margin-top:4px;"))
        self._lbl_disease = QLabel("—", alignment=Qt.AlignmentFlag.AlignLeft)
        self._lbl_disease.setFixedHeight(26)
        self._set_disease_style("", "")
        il.addWidget(self._lbl_disease)
        layout.addWidget(info_frame)

        # 裁剪信息（裁剪模式时显示）
        self._crop_frame = QFrame()
        self._crop_frame.setStyleSheet("border-bottom: 1px solid #eee; background:#fffbe6;")
        cl = QVBoxLayout(self._crop_frame)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(3)
        cl.addWidget(QLabel("裁剪区域", styleSheet="color:#b45309;font-size:10px;font-weight:bold;"))
        self._lbl_crop = QLabel("—", styleSheet="color:#555;font-size:11px;")
        cl.addWidget(self._lbl_crop)
        self._crop_frame.setVisible(False)
        layout.addWidget(self._crop_frame)

        # 区域图例
        legend_frame = QFrame()
        ll = QVBoxLayout(legend_frame)
        ll.setContentsMargins(12, 10, 12, 10)
        ll.setSpacing(4)
        self._legend_title = QLabel("标注区域 (0)", styleSheet="color:#999;font-size:10px;")
        ll.addWidget(self._legend_title)
        self._legend_list = QListWidget()
        self._legend_list.setStyleSheet("""
            QListWidget { border:none; background:transparent; font-size:11px; }
            QListWidget::item { padding:4px 0; border-bottom:1px solid #f5f5f5; }
        """)
        self._legend_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ll.addWidget(self._legend_list)
        legend_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(legend_frame, stretch=1)

    def _row(self, parent_layout, label_text):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(QLabel(label_text, styleSheet="color:#999;min-width:44px;"))
        val = QLabel("—", styleSheet="color:#222;font-weight:500;")
        val.setWordWrap(True)
        h.addWidget(val, stretch=1)
        parent_layout.addWidget(row)
        return val

    def _set_disease_style(self, disease: str, text: str):
        bg, fg = DISEASE_BADGE_COLORS.get(disease, ("#eee", "#555"))
        self._lbl_disease.setText(text or "—")
        self._lbl_disease.setStyleSheet(
            f"font-weight:bold;font-size:12px;padding:3px 10px;"
            f"border-radius:10px;background:{bg};color:{fg};margin-top:2px;"
        )

    def update_info(self, filename, size, patient, disease, labels, color_map):
        self._lbl_filename.setText(filename)
        self._lbl_size.setText(f"{size[0]} × {size[1]}" if size else "—")
        self._lbl_patient.setText(patient)
        self._set_disease_style(disease, disease)
        self._legend_list.clear()
        self._legend_title.setText(f"标注区域 ({len(labels)})")
        for label in labels:
            color = color_map.get(label, QColor(200, 200, 200))
            item = QListWidgetItem(f"  {label}")
            sw = QPixmap(14, 14); sw.fill(color)
            item.setIcon(QIcon(sw))
            self._legend_list.addItem(item)

    def update_crop_info(self, rect: QRectF | None):
        if rect and rect.isValid() and rect.width() > 0:
            self._crop_frame.setVisible(True)
            self._lbl_crop.setText(
                f"X: {int(rect.x())}  Y: {int(rect.y())}\n"
                f"W: {int(rect.width())}  H: {int(rect.height())}"
            )
        else:
            self._crop_frame.setVisible(False)

    def clear_info(self):
        self._lbl_filename.setText("—")
        self._lbl_size.setText("—")
        self._lbl_patient.setText("—")
        self._set_disease_style("", "")
        self._legend_list.clear()
        self._legend_title.setText("标注区域 (0)")
        self._crop_frame.setVisible(False)


# ── 主窗口 ────────────────────────────────────────────────────────────────────

BTN_ACTIVE = (
    "QPushButton{background:#3a6fd8;color:white;border:1px solid #3a6fd8;"
    "border-radius:4px;padding:2px 8px;font-size:12px;}"
)
BTN_NORMAL = (
    "QPushButton{background:transparent;color:#555;border:1px solid #ccc;"
    "border-radius:4px;padding:2px 8px;font-size:12px;}"
    "QPushButton:hover{background:#f0f0f0;}"
)
BTN_WARN = (
    "QPushButton{background:#b45309;color:white;border:1px solid #b45309;"
    "border-radius:4px;padding:2px 8px;font-size:12px;}"
    "QPushButton:hover{background:#92400e;}"
)


class MainWindow(QMainWindow):
    def __init__(self, color_map: dict[str, QColor]):
        super().__init__()
        self.color_map = color_map
        self.setWindowTitle("膝关节超声标注查看器")
        self.resize(1280, 780)

        toolbar = QToolBar("工具栏", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        def tbtn(text, width=None):
            b = QPushButton(text)
            b.setFixedHeight(28)
            if width:
                b.setFixedWidth(width)
            toolbar.addWidget(b)
            return b

        self._btn_ann  = tbtn("标注 ON", 72); self._btn_ann.setCheckable(True);  self._btn_ann.setChecked(True)
        self._btn_lbl  = tbtn("名称 ON", 72); self._btn_lbl.setCheckable(True);  self._btn_lbl.setChecked(True)
        tbtn("－").clicked.connect(lambda: self._viewer.zoom_out())
        tbtn("适配").clicked.connect(lambda: self._viewer.zoom_reset())
        tbtn("＋").clicked.connect(lambda: self._viewer.zoom_in())

        # 分隔
        sep = QWidget(); sep.setFixedWidth(10); toolbar.addWidget(sep)

        self._btn_crop   = tbtn("✂ 裁剪  [空格]", 110)
        self._btn_cancel = tbtn("✕ 取消", 70)
        self._btn_cancel.setVisible(False)
        self._in_crop_mode = False

        sep2 = QWidget(); sep2.setFixedWidth(10); toolbar.addWidget(sep2)

        self._btn_undo = tbtn("↩ 撤回", 72)
        self._btn_redo = tbtn("↪ 恢复", 72)
        self._btn_undo.setEnabled(False)
        self._btn_redo.setEnabled(False)

        self._undo_cache: dict | None = None   # 上一次裁剪的缓存
        self._redo_cache: dict | None = None   # 撤回后可恢复的缓存

        self._btn_ann.clicked.connect(self._toggle_ann)
        self._btn_lbl.clicked.connect(self._toggle_lbl)
        self._btn_crop.clicked.connect(self._on_crop_btn)
        self._btn_cancel.clicked.connect(self._cancel_crop)
        self._btn_undo.clicked.connect(self._do_undo)
        self._btn_redo.clicked.connect(self._do_redo)

        # 空格快捷键：窗口级别，焦点在任何子控件时均生效
        space_sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        space_sc.activated.connect(self._on_crop_btn)

        self._refresh_styles()

        # 三栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        self._tree = FileTree()
        self._tree.setMinimumWidth(180); self._tree.setMaximumWidth(300)
        self._tree.populate(DATA_ROOT)
        self._tree.itemClicked.connect(self._on_tree_click)
        splitter.addWidget(self._tree)

        self._viewer = ImageViewer()
        self._viewer.crop_rect_changed.connect(self._on_crop_changed)
        splitter.addWidget(self._viewer)

        self._panel = InfoPanel()
        splitter.addWidget(self._panel)
        splitter.setSizes([220, 820, 240])
        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self._status.setStyleSheet("font-size:11px;color:#666;")
        self.setStatusBar(self._status)
        self._status.showMessage("请从左侧文件树选择一张图片")

    # ── 文件树点击 ──

    def _on_tree_click(self, item: QTreeWidgetItem, _col: int):
        img_path: Path | None = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(img_path, Path):
            return
        disease    = img_path.parent.parent.name
        patient_raw = img_path.parent.name
        patient    = patient_raw.split("_", 1)[-1] if "_" in patient_raw else patient_raw
        pixmap     = QPixmap(str(img_path))
        size       = (pixmap.width(), pixmap.height()) if not pixmap.isNull() else None
        labels     = self._viewer.load_image(img_path, self.color_map)
        self._panel.update_info(img_path.name, size, patient, disease, labels, self.color_map)
        self._status.showMessage(f"{disease} / {patient} / {img_path.name}   |   {len(labels)} 个标注区域")
        # 重置裁剪按钮状态
        self._in_crop_mode = False
        self._btn_crop.setText("✂ 裁剪  [空格]")
        self._btn_cancel.setVisible(False)
        self._refresh_styles()

    # ── 工具栏 ──

    def _toggle_ann(self):
        show = self._btn_ann.isChecked()
        self._btn_ann.setText("标注 ON" if show else "标注 OFF")
        self._viewer.toggle_annotations(show)
        self._refresh_styles()

    def _toggle_lbl(self):
        show = self._btn_lbl.isChecked()
        self._btn_lbl.setText("名称 ON" if show else "名称 OFF")
        self._viewer.toggle_labels(show)
        self._refresh_styles()

    def _on_crop_btn(self):
        """裁剪按钮：第一次点击进入裁剪模式，第二次点击确认保存。"""
        if not self._in_crop_mode:
            if self._viewer.get_image_path() is None:
                self._status.showMessage("请先选择一张图片")
                return
            self._in_crop_mode = True
            self._btn_crop.setText("✓ 确认裁剪  [空格]")
            self._btn_cancel.setVisible(True)
            self._viewer.enter_crop_mode()
            self._status.showMessage("拖拽画出裁剪区域，8个控制点可调整边界 | 空格/再次点击确认  ESC/取消按钮退出")
        else:
            self._do_save_crop()
        self._refresh_styles()

    def _cancel_crop(self):
        self._in_crop_mode = False
        self._btn_crop.setText("✂ 裁剪  [空格]")
        self._btn_cancel.setVisible(False)
        self._viewer.cancel_crop()
        self._panel.update_crop_info(None)
        self._refresh_styles()
        self._status.showMessage("已取消裁剪")

    def _on_crop_changed(self, rect: QRectF):
        self._panel.update_crop_info(rect if rect.isValid() else None)
        if rect.isValid() and rect.width() > 0:
            self._status.showMessage(
                f"裁剪区域  X:{int(rect.x())} Y:{int(rect.y())}  "
                f"W:{int(rect.width())} H:{int(rect.height())}   |   Enter/保存裁剪 确认  ESC 取消"
            )

    def _do_save_crop(self):
        crop_rect = self._viewer.get_crop_rect()
        if crop_rect is None:
            self._status.showMessage("请先框选裁剪区域")
            return
        img_path = self._viewer.get_image_path()
        if img_path is None:
            return

        # 限制裁剪框在图片范围内
        w, h = self._viewer._img_size
        crop_rect = crop_rect.intersected(QRectF(0, 0, w, h))
        if crop_rect.width() < 4 or crop_rect.height() < 4:
            self._status.showMessage("裁剪区域太小，请重新选择")
            return

        try:
            cache = save_crop(img_path, crop_rect)
            self._undo_cache = cache
            self._redo_cache = None
            self._btn_undo.setEnabled(True)
            self._btn_redo.setEnabled(False)

            ann_count = len(json.loads(cache["new_json"]).get("objects", [])) if cache["new_json"] else 0
            self._cancel_crop()
            self._reload_image(img_path)
            self._status.showMessage(
                f"裁剪完成  {int(crop_rect.width())} × {int(crop_rect.height())}  "
                f"保留标注 {ann_count} 个  |  {img_path.name}   [↩ 可撤回]"
            )
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _reload_image(self, img_path: Path):
        """重新加载图片并刷新右侧面板。"""
        labels  = self._viewer.load_image(img_path, self.color_map)
        disease = img_path.parent.parent.name
        patient = img_path.parent.name.split("_", 1)[-1]
        pixmap  = QPixmap(str(img_path))
        size    = (pixmap.width(), pixmap.height()) if not pixmap.isNull() else None
        self._panel.update_info(img_path.name, size, patient, disease, labels, self.color_map)

    def _do_undo(self):
        if not self._undo_cache:
            return
        c = self._undo_cache
        c["path"].write_bytes(c["orig_img"])
        if c["orig_json"] is not None:
            c["path"].with_suffix(".json").write_text(c["orig_json"], encoding="utf-8")
        self._redo_cache = c
        self._undo_cache = None
        self._btn_undo.setEnabled(False)
        self._btn_redo.setEnabled(True)
        self._reload_image(c["path"])
        self._status.showMessage(f"已撤回裁剪  |  {c['path'].name}   [↪ 可恢复]")

    def _do_redo(self):
        if not self._redo_cache:
            return
        c = self._redo_cache
        c["path"].write_bytes(c["new_img"])
        if c["new_json"] is not None:
            c["path"].with_suffix(".json").write_text(c["new_json"], encoding="utf-8")
        self._undo_cache = c
        self._redo_cache = None
        self._btn_undo.setEnabled(True)
        self._btn_redo.setEnabled(False)
        self._reload_image(c["path"])
        self._status.showMessage(f"已恢复裁剪  |  {c['path'].name}   [↩ 可撤回]")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._in_crop_mode:
                self._do_save_crop()
                return
        super().keyPressEvent(event)

    def _refresh_styles(self):
        self._btn_ann.setStyleSheet(BTN_ACTIVE if self._btn_ann.isChecked() else BTN_NORMAL)
        self._btn_lbl.setStyleSheet(BTN_ACTIVE if self._btn_lbl.isChecked() else BTN_NORMAL)
        self._btn_crop.setStyleSheet(BTN_WARN if self._in_crop_mode else BTN_NORMAL)
        self._btn_cancel.setStyleSheet(BTN_NORMAL)
        self._btn_undo.setStyleSheet(BTN_NORMAL)
        self._btn_redo.setStyleSheet(BTN_NORMAL)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    print("正在加载标注类别...")
    all_labels = collect_all_labels()
    color_map = build_color_map(all_labels)
    print(f"就绪：{len(all_labels)} 个标注类别")
    window = MainWindow(color_map)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
