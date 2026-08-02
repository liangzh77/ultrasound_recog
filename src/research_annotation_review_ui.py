"""Diagnosis-blinded, ROI-only preview UI for the clinical review queue."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


WINDOW_STYLE = """
QMainWindow { background: #f5f6f7; color: #1f2933; }
QFrame#header { background: #ffffff; border-bottom: 1px solid #d9dee3; }
QLabel#title { font-size: 20px; font-weight: 600; color: #17212b; }
QLabel#status { color: #7a4b00; background: #fff4d6; border: 1px solid #e8c568; padding: 6px 10px; }
QLabel#target { font-size: 24px; font-weight: 600; color: #17212b; }
QLabel#meta { color: #52606d; }
QLabel#image { background: #111820; color: #f5f7fa; border: 1px solid #303b46; padding: 8px; }
QPushButton { min-height: 36px; padding: 0 16px; border: 1px solid #aeb7c0; background: #ffffff; color: #17212b; }
QPushButton:hover { background: #eef2f5; }
QPushButton:focus { border: 2px solid #1f6feb; }
QPushButton:disabled { color: #9aa5b1; background: #eef0f2; }
"""


def configure_cjk_font(application) -> str | None:
    """Install an available CJK font explicitly, including offscreen Qt runs."""
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = (
        windows_root / "Fonts" / "msyh.ttc",
        windows_root / "Fonts" / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QFont(families[0], 10))
            return families[0]
    return None


def load_roi_qimage(image_path: Path, annotation_path: Path) -> QImage:
    """Read only the confirmed ROI into a detached RGB QImage."""
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图像无法读取")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    roi = annotation.get("ultrasound_rect") or {}
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(round(float(roi.get("x1", 0))))))
    y1 = max(0, min(height, int(round(float(roi.get("y1", 0))))))
    x2 = max(0, min(width, int(round(float(roi.get("x2", width))))))
    y2 = max(0, min(height, int(round(float(roi.get("y2", height))))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("人工确认ROI无效")
    cropped = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    qimage = QImage(
        cropped.data,
        cropped.shape[1],
        cropped.shape[0],
        cropped.strides[0],
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()


class ReviewImageRepository:
    """Resolve pseudonymous image keys without exporting private linkage."""

    def __init__(self, root: Path, source_rows: list[Mapping[str, Any]]) -> None:
        self.root = root.resolve()
        self._sources = {
            str(row["image_key"]): {
                "image": str(row.get("raw_image_path", "")),
                "annotation": str(row.get("normalized_annotation_path", "")),
            }
            for row in source_rows
        }

    def load_roi(self, image_key: str) -> QImage:
        source = self._sources.get(image_key)
        if not source:
            raise ValueError("伪匿名图像键没有内部关联")
        image_path = (self.root / source["image"]).resolve()
        annotation_path = (self.root / source["annotation"]).resolve()
        for path in (image_path, annotation_path):
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise ValueError("内部关联超出项目目录") from error
            if not path.is_file():
                raise ValueError("内部关联文件缺失")
        return load_roi_qimage(image_path, annotation_path)


def audit_review_queue_rois(
    rows: list[Mapping[str, Any]], repository: ReviewImageRepository
) -> dict[str, Any]:
    """Load every queue ROI and return privacy-safe aggregate evidence."""
    widths = []
    heights = []
    failures = []
    for row in rows:
        try:
            image = repository.load_roi(str(row["image_key"]))
            widths.append(image.width())
            heights.append(image.height())
        except Exception:
            failures.append(str(row["review_case_key"]))
    return {
        "rows": len(rows),
        "loaded_rois": len(widths),
        "failed_rois": len(failures),
        "failed_review_case_keys": failures,
        "roi_width_min_max": [min(widths), max(widths)] if widths else None,
        "roi_height_min_max": [min(heights), max(heights)] if heights else None,
        "diagnosis_or_legacy_annotation_loaded_for_display": False,
        "pixels_outside_confirmed_roi_loaded_for_display": False,
    }


class ReviewQueuePreviewWindow(QMainWindow):
    """Read-only queue preview that never displays diagnoses or legacy labels."""

    def __init__(
        self,
        rows: list[Mapping[str, Any]],
        repository: ReviewImageRepository,
    ) -> None:
        super().__init__()
        if not rows:
            raise ValueError("复核队列为空")
        self._rows = [dict(row) for row in rows]
        self._repository = repository
        self._index = 0
        self._source_pixmap = QPixmap()
        self.setWindowTitle("区域/病变临床复核 · 草案预览")
        self.setStyleSheet(WINDOW_STYLE)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        title = QLabel("区域/病变临床复核")
        title.setObjectName("title")
        status = QLabel("草案只读预览 · 不记录结果 · 不用于训练")
        status.setObjectName("status")
        status.setAccessibleName("草案只读状态")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(status)
        layout.addWidget(header)

        self._target = QLabel()
        self._target.setObjectName("target")
        self._target.setAccessibleName("当前复核目标")
        self._meta = QLabel()
        self._meta.setObjectName("meta")
        self._meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._target)
        layout.addWidget(self._meta)

        self._image = QLabel("正在读取ROI…")
        self._image.setObjectName("image")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(640, 480)
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image, 1)

        helper = QLabel(
            "仅显示人工确认ROI，不显示诊断、旧多边形、原文件名或路径。"
            "使用左右方向键浏览。"
        )
        helper.setObjectName("meta")
        helper.setWordWrap(True)
        layout.addWidget(helper)

        controls = QHBoxLayout()
        self._previous = QPushButton("← 上一张")
        self._next = QPushButton("下一张 →")
        self._previous.setAccessibleName("上一张复核图像")
        self._next.setAccessibleName("下一张复核图像")
        self._previous.clicked.connect(lambda: self.navigate(-1))
        self._next.clicked.connect(lambda: self.navigate(1))
        controls.addWidget(self._previous)
        controls.addStretch(1)
        controls.addWidget(self._next)
        layout.addLayout(controls)

        self.setCentralWidget(root)
        self._shortcuts = [
            QShortcut(
                QKeySequence(Qt.Key.Key_Left),
                self,
                activated=lambda: self.navigate(-1),
            ),
            QShortcut(
                QKeySequence(Qt.Key.Key_Right),
                self,
                activated=lambda: self.navigate(1),
            ),
        ]
        self.setMinimumSize(960, 680)
        self.resize(1180, 780)
        self.show_index(0)

    @property
    def current_index(self) -> int:
        return self._index

    def navigate(self, delta: int) -> None:
        self.show_index(max(0, min(len(self._rows) - 1, self._index + delta)))

    def show_index(self, index: int) -> None:
        self._index = index
        row = self._rows[index]
        case_key = str(row["review_case_key"])
        target = str(row["target_category"])
        self._target.setText(f"复核目标：{target}")
        self._meta.setText(f"{index + 1} / {len(self._rows)}　病例：{case_key}")
        self._previous.setEnabled(index > 0)
        self._next.setEnabled(index < len(self._rows) - 1)
        try:
            image = self._repository.load_roi(str(row["image_key"]))
            self._source_pixmap = QPixmap.fromImage(image)
            self._image.setAccessibleName(f"{case_key}的ROI超声图像，目标{target}")
            self._render_pixmap()
        except Exception as error:
            self._source_pixmap = QPixmap()
            self._image.setPixmap(QPixmap())
            self._image.setText(f"无法显示当前ROI：{error}")
            self._image.setAccessibleName("当前ROI读取失败")

    def _render_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        available = self._image.size()
        self._image.setPixmap(
            self._source_pixmap.scaled(
                available,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render_pixmap()
