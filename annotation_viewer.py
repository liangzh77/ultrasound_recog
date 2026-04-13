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

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsPolygonItem, QGraphicsPixmapItem, QGraphicsTextItem,
    QGraphicsRectItem, QGraphicsPathItem,
    QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QToolBar, QStatusBar, QFrame, QPushButton, QSizePolicy, QDoubleSpinBox,
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
CANDIDATE_CLICK_BAND = 54


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


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def normalize_profile(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def estimate_foreground_range(profile: np.ndarray, active_ratio: float = 0.1) -> tuple[int, int]:
    if len(profile) == 0:
        raise ValueError("空轮廓，无法估计前景范围")
    positive = profile[profile > 0]
    if len(positive) == 0:
        raise ValueError("图像几乎全黑，无法自动框选")

    percentile = max(0.0, min(100.0, (1.0 - active_ratio) * 100.0))
    threshold = float(np.percentile(positive, percentile))
    active = profile >= threshold
    runs = find_runs(active)
    if not runs:
        raise ValueError("未找到稳定前景区域")

    gap_limit = max(6, len(profile) // 80)
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = runs[0]
    for start, end in runs[1:]:
        if start - cur_end <= gap_limit:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    center = (len(profile) - 1) * 0.5
    start, end = max(
        merged,
        key=lambda run: float(profile[run[0]:run[1] + 1].sum()) - abs(((run[0] + run[1]) * 0.5) - center) * threshold * 0.1,
    )
    return start, end


def scan_edge_by_jump(
    profile: np.ndarray,
    start: int,
    stop: int,
    step: int,
    jump_ratio: float,
    normalize_by: int,
) -> int:
    prev = int(profile[start])
    last_hit: int | None = None
    best_idx = start
    best_delta = float("-inf")
    scale = max(int(normalize_by), 1)

    for idx in range(start + step, stop, step):
        cur = int(profile[idx])
        delta = cur - prev
        ratio = delta / scale
        if ratio >= jump_ratio:
            last_hit = idx
        if delta > best_delta:
            best_delta = float(delta)
            best_idx = idx
        prev = cur

    return last_hit if last_hit is not None else best_idx


def collect_edge_candidates(
    profile: np.ndarray,
    start: int,
    stop: int,
    step: int,
    jump_ratio: float,
    normalize_by: int,
    fallback_idx: int,
) -> list[int]:
    if len(profile) == 0:
        return [fallback_idx]
    prev = int(profile[start])
    scale = max(int(normalize_by), 1)
    candidates: list[int] = []
    deltas: list[tuple[float, int]] = []
    for idx in range(start + step, stop, step):
        cur = int(profile[idx])
        delta = cur - prev
        ratio = delta / scale
        if ratio >= jump_ratio:
            candidates.append(idx)
        if delta > 0:
            deltas.append((float(delta), idx))
        prev = cur

    # 阈值命中的点太少时，补充若干个局部跳变最强的位置，保证可切换候选边界。
    if len(candidates) < 3 and deltas:
        min_gap = max(6, len(profile) // 40)
        for _, idx in sorted(deltas, key=lambda item: item[0], reverse=True):
            if all(abs(idx - existing) >= min_gap for existing in candidates):
                candidates.append(idx)
            if len(candidates) >= 8:
                break

    if fallback_idx not in candidates:
        candidates.append(fallback_idx)
    return sorted(set(candidates))


def pick_central_run(profile: np.ndarray, min_len: int, center: float) -> tuple[int, int] | None:
    active = profile >= 0.28
    runs = [run for run in find_runs(active) if run[1] - run[0] + 1 >= min_len]
    if not runs:
        active = profile >= 0.18
        runs = [run for run in find_runs(active) if run[1] - run[0] + 1 >= max(8, min_len // 2)]
    if not runs:
        return None
    return max(
        runs,
        key=lambda run: (run[1] - run[0] + 1) - abs(((run[0] + run[1]) * 0.5) - center) * 0.45,
    )


def refine_edge(profile: np.ndarray, anchor: int, prefer_rising: bool, radius: int) -> int:
    if len(profile) < 3:
        return anchor
    left = max(1, anchor - radius)
    right = min(len(profile) - 2, anchor + radius)
    if right < left:
        return anchor
    grad = np.diff(profile)
    segment = grad[left - 1:right]
    offset = int(np.argmax(segment) if prefer_rising else np.argmin(segment))
    return left + offset


def detect_ultrasound_geometry(image_path: Path, jump_ratio: float = 0.25) -> dict:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ValueError("图片尺寸过小，无法自动框选")

    mask = np.any(image >= 2, axis=2).astype(np.int32)
    center_x0 = int(w * 0.20)
    center_x1 = max(center_x0 + 1, int(w * 0.80))
    center_y0 = int(h * 0.20)
    center_y1 = max(center_y0 + 1, int(h * 0.80))

    center_mask = mask[center_y0:center_y1, center_x0:center_x1]
    row_profile_center = center_mask.sum(axis=1)
    col_profile_center = center_mask.sum(axis=0)

    rough_top_rel, rough_bottom_rel = estimate_foreground_range(row_profile_center, active_ratio=0.1)
    rough_left_rel, rough_right_rel = estimate_foreground_range(col_profile_center, active_ratio=0.1)
    rough_top = center_y0 + rough_top_rel
    rough_bottom = center_y0 + rough_bottom_rel
    rough_left = center_x0 + rough_left_rel
    rough_right = center_x0 + rough_right_rel

    edge_exclude_x = int(w * 0.10)
    row_x0 = max(edge_exclude_x, rough_left)
    row_x1 = min(w - edge_exclude_x - 1, rough_right)
    if row_x1 <= row_x0:
        row_x0 = rough_left
        row_x1 = rough_right

    row_profile = mask[:, row_x0:row_x1 + 1].sum(axis=1)
    col_profile = mask[rough_top:rough_bottom + 1, :].sum(axis=0)

    fg_top, fg_bottom = estimate_foreground_range(row_profile, active_ratio=0.1)
    fg_left, fg_right = estimate_foreground_range(col_profile, active_ratio=0.1)

    edge_exclude_scan_x = int(w * 0.05)
    left_start = min(edge_exclude_scan_x, max(0, w - 2))
    right_start = max(0, w - edge_exclude_scan_x - 1)

    left = scan_edge_by_jump(col_profile, left_start, fg_left, 1, jump_ratio, h)
    right = scan_edge_by_jump(col_profile, right_start, fg_right, -1, jump_ratio, h)
    top = scan_edge_by_jump(row_profile, 0, fg_top, 1, jump_ratio, w)
    bottom = scan_edge_by_jump(row_profile, h - 1, fg_bottom, -1, jump_ratio, w)

    candidate_margin_x = max(8, int(w * 0.08))
    candidate_margin_y = max(8, int(h * 0.08))
    left_candidate_stop = min(w - 1, fg_left + candidate_margin_x)
    right_candidate_stop = max(0, fg_right - candidate_margin_x)
    top_candidate_stop = min(h - 1, fg_top + candidate_margin_y)
    bottom_candidate_stop = max(0, fg_bottom - candidate_margin_y)

    left_candidates = collect_edge_candidates(col_profile, left_start, left_candidate_stop, 1, jump_ratio, h, left)
    right_candidates = collect_edge_candidates(col_profile, right_start, right_candidate_stop, -1, jump_ratio, h, right)
    top_candidates = collect_edge_candidates(row_profile, 0, top_candidate_stop, 1, jump_ratio, w, top)
    bottom_candidates = collect_edge_candidates(row_profile, h - 1, bottom_candidate_stop, -1, jump_ratio, w, bottom)

    top = max(0, min(top, h - 2))
    bottom = max(top + 2, min(bottom + 1, h))
    left = max(0, min(left, w - 2))
    right = max(left + 2, min(right + 1, w))

    rect = QRectF(float(left), float(top), float(right - left), float(bottom - top))
    if rect.width() < 32 or rect.height() < 32:
        raise ValueError("自动框选结果过小，请手动调整")
    return {
        "rect": rect,
        "candidates": {
            "left": [int(v) for v in left_candidates if 0 <= v < w],
            "right": [int(v) for v in right_candidates if 0 <= v < w],
            "top": [int(v) for v in top_candidates if 0 <= v < h],
            "bottom": [int(v) for v in bottom_candidates if 0 <= v < h],
        },
    }


def detect_ultrasound_rect(image_path: Path, jump_ratio: float = 0.25) -> QRectF:
    return detect_ultrasound_geometry(image_path, jump_ratio)["rect"]


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
    仅更新标注 JSON，记录超声图像矩形区域，不修改原图和分割点。
    返回 undo/redo 缓存字典：
      {path, orig_img, orig_json, new_img, new_json}
    其中图片字段保留为 None，仅为了复用现有 undo/redo 结构。
    """
    x1 = int(round(crop_rect.x()))
    y1 = int(round(crop_rect.y()))
    x2 = int(round(crop_rect.x() + crop_rect.width()))
    y2 = int(round(crop_rect.y() + crop_rect.height()))
    json_path = image_path.with_suffix(".json")

    # ── 先读原始内容（用于 undo）──
    orig_json_str  = json_path.read_text(encoding="utf-8") if json_path.exists() else None

    # ── 更新标注 JSON（覆盖原 JSON）──
    new_json_str = None
    if orig_json_str is not None:
        data     = json.loads(orig_json_str)
        new_data = copy.deepcopy(data)
        new_data["ultrasound_rect"] = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        new_data["cropped"] = False
        new_json_str = json.dumps(new_data, ensure_ascii=False, indent=4)
        json_path.write_text(new_json_str, encoding="utf-8")

    return {
        "path":      image_path,
        "orig_img":  None,
        "orig_json": orig_json_str,
        "new_img":   None,
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
        self._edge_hotspots: dict[str, QGraphicsPathItem] = {}
        self._img_rect = QRectF()
        self._crop_rect = QRectF()
        self._hover_edge: str | None = None

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

        for edge in ("left", "right", "top", "bottom"):
            item = QGraphicsPathItem()
            item.setPen(QPen(Qt.PenStyle.NoPen))
            item.setBrush(QBrush(QColor(59, 130, 246, 0)))
            item.setZValue(10.5)
            self._scene.addItem(item)
            self._edge_hotspots[edge] = item

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
        for item in [self._overlay, self._border] + list(self._handles.values()) + list(self._edge_hotspots.values()):
            if item and item.scene():
                self._scene.removeItem(item)
        self._overlay = None
        self._border = None
        self._handles.clear()
        self._edge_hotspots.clear()
        self._crop_rect = QRectF()
        self._hover_edge = None

    def set_rect(self, rect: QRectF):
        self._crop_rect = rect.normalized()
        self._update_items()

    def get_rect(self) -> QRectF:
        return self._crop_rect.normalized()

    def is_valid(self) -> bool:
        r = self._crop_rect.normalized()
        return r.width() > 4 and r.height() > 4

    def _update_items(self, click_band: float = 0.0):
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

        if click_band > 0:
            self._update_hotspots(r, click_band)

    def _update_hotspots(self, r: QRectF, click_band: float):
        cx = r.center().x()
        cy = r.center().y()

        hotspot_paths: dict[str, QPainterPath] = {}

        left_path = QPainterPath()
        left_path.addRect(QRectF(self._img_rect.left(), self._img_rect.top(), max(1.0, r.left() - self._img_rect.left()), self._img_rect.height()))
        left_path.moveTo(r.left(), r.top())
        left_path.lineTo(cx, cy)
        left_path.lineTo(r.left(), r.bottom())
        left_path.closeSubpath()
        hotspot_paths["left"] = left_path

        right_path = QPainterPath()
        right_path.addRect(QRectF(r.right(), self._img_rect.top(), max(1.0, self._img_rect.right() - r.right()), self._img_rect.height()))
        right_path.moveTo(r.right(), r.top())
        right_path.lineTo(cx, cy)
        right_path.lineTo(r.right(), r.bottom())
        right_path.closeSubpath()
        hotspot_paths["right"] = right_path

        top_path = QPainterPath()
        top_path.addRect(QRectF(r.left(), self._img_rect.top(), max(1.0, r.width()), max(1.0, r.top() - self._img_rect.top())))
        top_path.moveTo(r.left(), r.top())
        top_path.lineTo(cx, cy)
        top_path.lineTo(r.right(), r.top())
        top_path.closeSubpath()
        hotspot_paths["top"] = top_path

        bottom_path = QPainterPath()
        bottom_path.addRect(QRectF(r.left(), r.bottom(), max(1.0, r.width()), max(1.0, self._img_rect.bottom() - r.bottom())))
        bottom_path.moveTo(r.left(), r.bottom())
        bottom_path.lineTo(cx, cy)
        bottom_path.lineTo(r.right(), r.bottom())
        bottom_path.closeSubpath()
        hotspot_paths["bottom"] = bottom_path

        for edge, item in self._edge_hotspots.items():
            item.setPath(hotspot_paths[edge])
            color = QColor(59, 130, 246, 80 if self._hover_edge == edge else 0)
            item.setBrush(QBrush(color))

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

    def set_edge(self, edge: str, value: float):
        r = self._crop_rect.normalized()
        if edge == "left":
            right = r.right()
            value = min(value, right - 4)
            self._crop_rect = QRectF(QPointF(value, r.top()), QPointF(right, r.bottom())).normalized()
        elif edge == "right":
            left = r.left()
            value = max(value, left + 4)
            self._crop_rect = QRectF(QPointF(left, r.top()), QPointF(value, r.bottom())).normalized()
        elif edge == "top":
            bottom = r.bottom()
            value = min(value, bottom - 4)
            self._crop_rect = QRectF(QPointF(r.left(), value), QPointF(r.right(), bottom)).normalized()
        elif edge == "bottom":
            top = r.top()
            value = max(value, top + 4)
            self._crop_rect = QRectF(QPointF(r.left(), top), QPointF(r.right(), value)).normalized()
        self._update_items()

    def set_hover_edge(self, edge: str | None, click_band: float):
        self._hover_edge = edge
        self._update_items(click_band)


# ── 图片查看器（含裁剪模式） ──────────────────────────────────────────────────

class CropMode(Enum):
    OFF      = 0
    DRAWING  = 1   # 正在拖拽画矩形
    ADJUSTING = 2  # 可拖动控制点或移动矩形


class ImageViewer(QGraphicsView):

    crop_rect_changed  = Signal(QRectF)   # 裁剪框变化时通知主窗口
    navigate_requested = Signal(int)      # -1=上一项, +1=下一项
    expand_requested   = Signal()         # 展开/折叠当前文件夹

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
        self._edge_candidates: dict[str, list[int]] = {"left": [], "right": [], "top": [], "bottom": []}

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

    def enter_crop_mode(self, initial_rect: QRectF | None = None):
        if self._pixmap_item is None:
            return
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        img_rect = QRectF(0, 0, *self._img_size)
        if initial_rect is not None and initial_rect.isValid() and initial_rect.width() > 0 and initial_rect.height() > 0:
            rect = initial_rect.intersected(img_rect)
            self._crop_overlay.activate(img_rect, rect)
            self._crop_overlay.set_hover_edge(None, self._candidate_click_band())
            self._crop_mode = CropMode.ADJUSTING
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.crop_rect_changed.emit(self._crop_overlay.get_rect())
        else:
            self._crop_mode = CropMode.DRAWING
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._crop_overlay.activate(img_rect)

    def _exit_crop_mode(self):
        self._crop_mode = CropMode.OFF
        self._drag_start = None
        self._drag_handle = None
        self._drag_move = False
        self._edge_candidates = {"left": [], "right": [], "top": [], "bottom": []}
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

    def set_edge_candidates(self, candidates: dict[str, list[int]] | None):
        self._edge_candidates = candidates or {"left": [], "right": [], "top": [], "bottom": []}

    # ── 控制点容差（从视口像素转场景坐标） ──

    def _handle_tol(self) -> float:
        inv, _ = self.transform().inverted()
        return inv.m11() * (HANDLE_SIZE * 0.8)

    def _candidate_click_band(self) -> float:
        inv, _ = self.transform().inverted()
        return inv.m11() * CANDIDATE_CLICK_BAND

    def _edge_hotspot_side_band(self, rect: QRectF, click_band: float) -> float:
        return max(click_band, min(rect.width() * 0.18, click_band * 4))

    def _edge_hotspot_top_band(self, rect: QRectF, click_band: float) -> float:
        return max(click_band, min(rect.height() * 0.18, click_band * 4))

    def _edge_at_click_zone(self, scene_pos: QPointF) -> str | None:
        rect = self._crop_overlay.get_rect()
        if not rect.isValid():
            return None
        img_rect = QRectF(0, 0, *self._img_size)
        x = scene_pos.x()
        y = scene_pos.y()
        if not img_rect.contains(scene_pos):
            return None

        # 矩形外：外侧区域直接对应最近的外边界
        if x < rect.left():
            return "left"
        if x > rect.right():
            return "right"
        if y < rect.top():
            return "top"
        if y > rect.bottom():
            return "bottom"

        # 矩形内：由两条对角线划分为 4 个三角区
        left = rect.left()
        top = rect.top()
        width = rect.width()
        height = rect.height()
        if width <= 0 or height <= 0:
            return None

        rel_x = (x - left) / width
        rel_y = (y - top) / height

        # 相对两条对角线的位置
        above_diag1 = rel_y <= rel_x           # y <= x
        above_diag2 = rel_y <= 1.0 - rel_x     # y <= 1-x

        if above_diag1 and above_diag2:
            return "top"
        if above_diag1 and not above_diag2:
            return "right"
        if not above_diag1 and above_diag2:
            return "left"
        return "bottom"
        return None

    def _refresh_hover_edge(self, scene_pos: QPointF | None):
        click_band = self._candidate_click_band()
        edge = self._edge_at_click_zone(scene_pos) if scene_pos is not None else None
        self._crop_overlay.set_hover_edge(edge, click_band)

    def _jump_edge_candidate(self, scene_pos: QPointF) -> bool:
        rect = self._crop_overlay.get_rect()
        if not rect.isValid():
            return False

        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        edge = self._edge_at_click_zone(scene_pos)
        if edge is None:
            return False

        def choose(candidates: list[int], current: float, toward_larger: bool) -> int | None:
            if not candidates:
                return None
            if toward_larger:
                larger = [c for c in candidates if c > current + 1]
                return min(larger) if larger else None
            smaller = [c for c in candidates if c < current - 1]
            return max(smaller) if smaller else None

        if edge == "left":
            new_left = choose(self._edge_candidates.get("left", []), left, toward_larger=scene_pos.x() > left)
            if new_left is not None:
                self._crop_overlay.set_edge("left", float(new_left))
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
                return True
        elif edge == "right":
            new_right = choose(self._edge_candidates.get("right", []), right, toward_larger=scene_pos.x() > right)
            if new_right is not None:
                self._crop_overlay.set_edge("right", float(new_right))
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
                return True
        elif edge == "top":
            new_top = choose(self._edge_candidates.get("top", []), top, toward_larger=scene_pos.y() > top)
            if new_top is not None:
                self._crop_overlay.set_edge("top", float(new_top))
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
                return True
        elif edge == "bottom":
            new_bottom = choose(self._edge_candidates.get("bottom", []), bottom, toward_larger=scene_pos.y() > bottom)
            if new_bottom is not None:
                self._crop_overlay.set_edge("bottom", float(new_bottom))
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
                return True

        return False

    # ── 鼠标事件 ──

    def mousePressEvent(self, event):
        if self._crop_mode == CropMode.OFF:
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if self._crop_mode == CropMode.DRAWING:
            self._drag_start = scene_pos
            self._crop_overlay.set_rect(QRectF(scene_pos, scene_pos))

        elif self._crop_mode == CropMode.ADJUSTING:
            handle = self._crop_overlay.handle_at(scene_pos, self._handle_tol())
            if handle:
                self._drag_handle = handle
                self.setCursor(CropOverlay.CURSORS[handle])
            elif not ctrl_pressed and self._jump_edge_candidate(scene_pos):
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self._refresh_hover_edge(scene_pos)
            elif ctrl_pressed and self._crop_overlay.get_rect().contains(scene_pos):
                self._drag_move = True
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif ctrl_pressed:
                # 在框外点击 → 重新画
                self._crop_mode = CropMode.DRAWING
                self._drag_start = scene_pos
                self._crop_overlay.set_rect(QRectF(scene_pos, scene_pos))
                self._refresh_hover_edge(None)
            else:
                self._refresh_hover_edge(scene_pos)
                self.setCursor(Qt.CursorShape.ArrowCursor)

        self._last_scene_pos = scene_pos

    def mouseMoveEvent(self, event):
        if self._crop_mode == CropMode.OFF:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if self._crop_mode == CropMode.DRAWING and self._drag_start:
            self._crop_overlay.set_rect(QRectF(self._drag_start, scene_pos))
            self.crop_rect_changed.emit(self._crop_overlay.get_rect())

        elif self._crop_mode == CropMode.ADJUSTING:
            if self._last_scene_pos is None:
                self._last_scene_pos = scene_pos
                return
            delta = scene_pos - self._last_scene_pos
            ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            if self._drag_handle:
                self._crop_overlay.apply_handle_drag(self._drag_handle, delta)
                self._refresh_hover_edge(scene_pos)
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
            elif self._drag_move:
                self._crop_overlay.move_rect(delta)
                self._refresh_hover_edge(scene_pos)
                self.crop_rect_changed.emit(self._crop_overlay.get_rect())
            else:
                # 悬停时更新光标
                handle = self._crop_overlay.handle_at(scene_pos, self._handle_tol())
                if handle:
                    self._refresh_hover_edge(None)
                    self.setCursor(CropOverlay.CURSORS[handle])
                elif ctrl_pressed and self._crop_overlay.get_rect().contains(scene_pos):
                    self._refresh_hover_edge(None)
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self._refresh_hover_edge(scene_pos)
                    self.setCursor(Qt.CursorShape.ArrowCursor)

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
        key = event.key()
        if key == Qt.Key.Key_Escape and self._crop_mode != CropMode.OFF:
            self._exit_crop_mode()
            self.crop_rect_changed.emit(QRectF())
        elif key == Qt.Key.Key_Up:
            self.navigate_requested.emit(-1)
        elif key == Qt.Key.Key_Down:
            self.navigate_requested.emit(1)
        elif key == Qt.Key.Key_Right:
            self.expand_requested.emit()
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
        self._filter_uncropped = False
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

    @staticmethod
    def _is_cropped(img_path: Path) -> bool:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            return False
        try:
            with open(json_path, encoding="utf-8") as f:
                d = json.load(f)
            return "ultrasound_rect" in d
        except Exception:
            return False

    def set_filter_uncropped(self, enabled: bool):
        """开启/关闭「仅显示未裁剪」过滤器，对已展开的节点立即生效。"""
        self._filter_uncropped = enabled
        root = self.invisibleRootItem()
        for di in range(root.childCount()):
            disease_item = root.child(di)
            for pi in range(disease_item.childCount()):
                patient_item = disease_item.child(pi)
                has_visible = False
                for ii in range(patient_item.childCount()):
                    img_item = patient_item.child(ii)
                    img_path = img_item.data(0, Qt.ItemDataRole.UserRole)
                    if isinstance(img_path, Path):
                        hide = enabled and self._is_cropped(img_path)
                        img_item.setHidden(hide)
                        if not hide:
                            has_visible = True
                # 患者文件夹：若无可见图片则隐藏
                if patient_item.childCount() > 0:
                    patient_item.setHidden(enabled and not has_visible)

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
                has_visible = False
                for img in sorted(f for f in patient_dir.iterdir()
                                  if f.suffix in IMAGE_EXTS and f.with_suffix(".json").exists()):
                    img_item = QTreeWidgetItem(p_item, [img.name])
                    img_item.setData(0, Qt.ItemDataRole.UserRole, img)
                    if self._filter_uncropped and self._is_cropped(img):
                        img_item.setHidden(True)
                    else:
                        has_visible = True
                if self._filter_uncropped and not has_visible:
                    p_item.setHidden(True)


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

        self._btn_crop = QPushButton("✂ 裁剪[回车]")
        self._btn_crop.setFixedHeight(28)
        self._btn_crop.setMinimumWidth(120)
        toolbar.addWidget(self._btn_crop)
        self._btn_auto_crop = tbtn("自动框选(\\)", 100)
        toolbar.addWidget(QLabel("跳变阈值"))
        self._spin_auto_jump = QDoubleSpinBox()
        self._spin_auto_jump.setDecimals(0)
        self._spin_auto_jump.setRange(5, 500)
        self._spin_auto_jump.setSingleStep(5)
        self._spin_auto_jump.setSuffix("%")
        self._spin_auto_jump.setValue(25)
        self._spin_auto_jump.setFixedHeight(28)
        self._spin_auto_jump.setFixedWidth(78)
        toolbar.addWidget(self._spin_auto_jump)
        self._btn_cancel = tbtn("✕ 取消", 70)
        self._btn_cancel.setVisible(False)
        self._in_crop_mode = False

        sep2 = QWidget(); sep2.setFixedWidth(10); toolbar.addWidget(sep2)

        self._btn_undo = tbtn("↩ 撤回", 72)
        self._btn_redo = tbtn("↪ 恢复", 72)
        self._btn_undo.setEnabled(False)
        self._btn_redo.setEnabled(False)

        sep3 = QWidget(); sep3.setFixedWidth(10); toolbar.addWidget(sep3)

        self._btn_filter = tbtn("仅未裁剪", 80)
        self._btn_filter.setCheckable(True)
        self._btn_filter.setChecked(False)

        self._undo_cache: dict | None = None   # 上一次裁剪的缓存
        self._redo_cache: dict | None = None   # 撤回后可恢复的缓存
        self._nav_direction: int = 0            # 导航方向：-1上 / 1下 / 0点击

        self._btn_ann.clicked.connect(self._toggle_ann)
        self._btn_lbl.clicked.connect(self._toggle_lbl)
        self._btn_crop.clicked.connect(self._on_crop_btn)
        self._btn_auto_crop.clicked.connect(self._on_auto_crop_btn)
        self._btn_cancel.clicked.connect(self._cancel_crop)
        self._btn_undo.clicked.connect(self._do_undo)
        self._btn_redo.clicked.connect(self._do_redo)
        self._btn_filter.clicked.connect(self._toggle_filter)

        # 回车快捷键：窗口级别，焦点在任何子控件时均生效
        enter_sc = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        enter_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        enter_sc.activated.connect(self._on_crop_btn)
        auto_crop_sc = QShortcut(QKeySequence("\\"), self)
        auto_crop_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        auto_crop_sc.activated.connect(self._on_auto_crop_btn)
        crop_next_sc = QShortcut(QKeySequence("'"), self)
        crop_next_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        crop_next_sc.activated.connect(self._confirm_crop_and_open_next_uncropped)

        self._refresh_styles()

        # 三栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        self._tree = FileTree()
        self._tree.setMinimumWidth(180); self._tree.setMaximumWidth(300)
        self._tree.populate(DATA_ROOT)
        self._tree.currentItemChanged.connect(self._on_tree_item_changed)
        splitter.addWidget(self._tree)

        self._viewer = ImageViewer()
        self._viewer.crop_rect_changed.connect(self._on_crop_changed)
        self._viewer.navigate_requested.connect(self._tree_navigate)
        self._viewer.expand_requested.connect(self._tree_expand)
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
        self._btn_crop.setText("✂ 裁剪[回车]")
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
            self._btn_crop.setText("✓ 确认裁剪[回车]")
            self._btn_cancel.setVisible(True)
            self._viewer.enter_crop_mode()
            self._status.showMessage("拖拽画出裁剪区域，8个控制点可调整边界 | 空格/再次点击确认  ESC/取消按钮退出")
        else:
            self._do_save_crop()
        self._refresh_styles()

    def _on_auto_crop_btn(self):
        img_path = self._viewer.get_image_path()
        if img_path is None:
            self._status.showMessage("请先选择一张图片")
            return
        try:
            jump_ratio = self._spin_auto_jump.value() / 100.0
            geometry = detect_ultrasound_geometry(img_path, jump_ratio=jump_ratio)
            crop_rect = geometry["rect"]
            self._in_crop_mode = True
            self._btn_crop.setText("✓ 确认裁剪[回车]")
            self._btn_cancel.setVisible(True)
            self._viewer.enter_crop_mode(crop_rect)
            self._viewer.set_edge_candidates(geometry.get("candidates"))
            self._panel.update_crop_info(crop_rect)
            self._status.showMessage(
                f"自动框选完成  X:{int(crop_rect.x())} Y:{int(crop_rect.y())}  "
                f"W:{int(crop_rect.width())} H:{int(crop_rect.height())}  阈值:{self._spin_auto_jump.value():.0f}%   |   可点击边界两侧切换候选边界，或拖动微调后确认"
            )
        except Exception as e:
            QMessageBox.warning(self, "自动框选失败", str(e))
            self._status.showMessage(f"自动框选失败：{e}")
        self._refresh_styles()

    def _cancel_crop(self):
        self._in_crop_mode = False
        self._btn_crop.setText("✂ 裁剪[回车]")
        self._btn_cancel.setVisible(False)
        self._viewer.set_edge_candidates(None)
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

    def _do_save_crop(self, open_next_uncropped: bool = False) -> bool:
        crop_rect = self._viewer.get_crop_rect()
        if crop_rect is None:
            self._status.showMessage("请先框选裁剪区域")
            return False
        img_path = self._viewer.get_image_path()
        if img_path is None:
            return False
        current_item = self._tree.currentItem()

        # 限制裁剪框在图片范围内
        w, h = self._viewer._img_size
        crop_rect = crop_rect.intersected(QRectF(0, 0, w, h))
        if crop_rect.width() < 4 or crop_rect.height() < 4:
            self._status.showMessage("裁剪区域太小，请重新选择")
            return False

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
                f"已记录超声区域  {int(crop_rect.width())} × {int(crop_rect.height())}  "
                f"图中标注 {ann_count} 个  |  {img_path.name}   [↩ 可撤回]"
            )
            # 过滤器开启时，裁剪完自动隐藏当前图片并跳到下一张
            if self._btn_filter.isChecked() and current_item:
                cur = self._tree.currentItem()
                if cur:
                    cur.setHidden(True)
                    parent = cur.parent()
                    if parent:
                        has_visible = any(
                            not parent.child(i).isHidden()
                            for i in range(parent.childCount())
                        )
                        if not has_visible:
                            parent.setHidden(True)
                if not open_next_uncropped:
                    self._tree_navigate(1)
            if open_next_uncropped and current_item:
                next_item = self._find_next_uncropped_item(current_item)
                if next_item:
                    self._tree.setCurrentItem(next_item)
                    self._tree.scrollToItem(next_item)
                else:
                    self._status.showMessage("裁剪完成，后面没有未裁剪图片了")
            return True
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return False

    def _confirm_crop_and_open_next_uncropped(self):
        if not self._do_save_crop(open_next_uncropped=True):
            return
        next_path = self._viewer.get_image_path()
        if next_path is not None and not self._tree._is_cropped(next_path):
            self._on_auto_crop_btn()

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
        if c["orig_img"] is not None:
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
        if c["new_img"] is not None:
            c["path"].write_bytes(c["new_img"])
        if c["new_json"] is not None:
            c["path"].with_suffix(".json").write_text(c["new_json"], encoding="utf-8")
        self._undo_cache = c
        self._redo_cache = None
        self._btn_undo.setEnabled(True)
        self._btn_redo.setEnabled(False)
        self._reload_image(c["path"])
        self._status.showMessage(f"已恢复裁剪  |  {c['path'].name}   [↩ 可撤回]")

    def _on_tree_item_changed(self, cur: QTreeWidgetItem, prev: QTreeWidgetItem):
        """当前选中项变化时触发：图片则加载，文件夹则自动展开并选边界图片。"""
        if cur is None:
            return
        img_path = cur.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(img_path, Path):
            # 图片节点 — 直接加载
            self._on_tree_click(cur, 0)
        else:
            # 文件夹节点 — 自动展开，按导航方向选第一张或最后一张
            if not cur.isExpanded():
                cur.setExpanded(True)
            if self._nav_direction == -1:
                target = self._last_image_child(cur)
            else:
                target = self._first_image_child(cur)
            if target:
                self._tree.setCurrentItem(target)
                self._tree.scrollToItem(target)

    def _tree_navigate(self, direction: int):
        """上下键导航文件树（从图像区域触发）。"""
        cur = self._tree.currentItem()
        nxt = self._tree.itemAbove(cur) if direction == -1 else self._tree.itemBelow(cur)
        if nxt is None:
            return
        # 按上键时，itemAbove 可能返回当前项的父文件夹（视觉上紧挨着）。
        # 这种情况下要跳过父节点，继续往上，否则会留在同一文件夹里。
        if direction == -1 and nxt is cur.parent():
            nxt = self._tree.itemAbove(nxt)
            if nxt is None:
                return
        self._nav_direction = direction
        self._tree.setCurrentItem(nxt)
        self._tree.scrollToItem(nxt)
        self._nav_direction = 0

    def _find_next_uncropped_item(self, start_item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        item = start_item
        while True:
            item = self._next_tree_item(item)
            if item is None:
                return None
            img_path = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(img_path, Path) and not self._tree._is_cropped(img_path):
                return item

    def _next_tree_item(self, item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        if item is None:
            return None

        if item.data(0, Qt.ItemDataRole.UserRole) is None:
            if not item.isExpanded():
                item.setExpanded(True)
            first_child = self._first_real_child(item)
            if first_child:
                return first_child

        sibling = self._next_sibling(item)
        if sibling:
            return sibling

        parent = item.parent()
        while parent is not None:
            sibling = self._next_sibling(parent)
            if sibling:
                return sibling
            parent = parent.parent()

        root = self._tree.invisibleRootItem()
        return self._next_top_level_after(item, root)

    @staticmethod
    def _first_real_child(item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        for i in range(item.childCount()):
            child = item.child(i)
            role = child.data(0, Qt.ItemDataRole.UserRole)
            if role == "__placeholder__":
                continue
            return child
        return None

    @staticmethod
    def _next_sibling(item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        parent = item.parent()
        if parent is None:
            return None
        index = parent.indexOfChild(item)
        for i in range(index + 1, parent.childCount()):
            sibling = parent.child(i)
            role = sibling.data(0, Qt.ItemDataRole.UserRole)
            if role == "__placeholder__":
                continue
            return sibling
        return None

    def _next_top_level_after(self, item: QTreeWidgetItem, root: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        cursor = item
        while cursor.parent() is not None:
            cursor = cursor.parent()
        index = root.indexOfChild(cursor)
        for i in range(index + 1, root.childCount()):
            sibling = root.child(i)
            first = self._first_real_child(sibling)
            if first is not None:
                return first
        return None

    def _first_image_child(self, item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        """递归找 item 下第一个图片节点（UserRole 是 Path 的节点）。"""
        for i in range(item.childCount()):
            child = item.child(i)
            if isinstance(child.data(0, Qt.ItemDataRole.UserRole), Path):
                return child
            # 子项也是文件夹，继续向下找
            child.setExpanded(True)
            found = self._first_image_child(child)
            if found:
                return found
        return None

    def _last_image_child(self, item: QTreeWidgetItem) -> "QTreeWidgetItem | None":
        """递归找 item 下最后一个图片节点。"""
        for i in range(item.childCount() - 1, -1, -1):
            child = item.child(i)
            if isinstance(child.data(0, Qt.ItemDataRole.UserRole), Path):
                return child
            child.setExpanded(True)
            found = self._last_image_child(child)
            if found:
                return found
        return None

    def _tree_expand(self):
        """右键展开当前文件夹，已展开则折叠。"""
        cur = self._tree.currentItem()
        if cur is None:
            return
        if cur.data(0, Qt.ItemDataRole.UserRole) is not None:
            return   # 图片节点，忽略
        cur.setExpanded(not cur.isExpanded())

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def _toggle_filter(self):
        enabled = self._btn_filter.isChecked()
        self._tree.set_filter_uncropped(enabled)
        self._refresh_styles()

    def _refresh_styles(self):
        self._btn_ann.setStyleSheet(BTN_ACTIVE if self._btn_ann.isChecked() else BTN_NORMAL)
        self._btn_lbl.setStyleSheet(BTN_ACTIVE if self._btn_lbl.isChecked() else BTN_NORMAL)
        self._btn_crop.setStyleSheet(BTN_WARN if self._in_crop_mode else BTN_NORMAL)
        self._btn_cancel.setStyleSheet(BTN_NORMAL)
        self._btn_undo.setStyleSheet(BTN_NORMAL)
        self._btn_redo.setStyleSheet(BTN_NORMAL)
        self._btn_filter.setStyleSheet(BTN_ACTIVE if self._btn_filter.isChecked() else BTN_NORMAL)


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
