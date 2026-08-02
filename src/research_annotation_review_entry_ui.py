"""Diagnosis-blinded formal S1a semantic review entry window."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.research_annotation_review_entry import (
    REQUIRED_SEMANTIC_FIELDS,
    REVIEWER_VALUE_FIELDS,
    reviewer_prefix,
    validate_reviewer_response_rows,
)
from src.research_annotation_review_ui import ReviewImageRepository, WINDOW_STYLE


ENTRY_STYLE = WINDOW_STYLE + """
QFrame#formPanel { background: #ffffff; border: 1px solid #d9dee3; }
QLabel#sectionTitle { font-size: 17px; font-weight: 600; color: #17212b; }
QLabel#entryStatus { color: #334e68; background: #eaf4fb; border: 1px solid #9fbfd3; padding: 7px 10px; }
QComboBox, QPlainTextEdit { border: 1px solid #aeb7c0; background: #ffffff; color: #17212b; padding: 5px; }
QComboBox:focus, QPlainTextEdit:focus { border: 2px solid #1f6feb; }
QProgressBar { border: 1px solid #aeb7c0; background: #eef0f2; text-align: center; min-height: 22px; }
QProgressBar::chunk { background: #2f6f4e; }
"""

VALUE_LABELS = {
    "present": "明确存在",
    "absent_visible": "可见范围内明确不存在",
    "not_in_view": "目标不在当前切面/可见范围",
    "uncertain": "无法确定",
    "B": "灰阶 B 模式",
    "PD": "能量多普勒 PD",
    "CD": "彩色多普勒 CD",
    "unknown": "模式不确定",
    "exhaustive_for_declared_targets": "对声明目标作穷尽判断",
    "positive_only": "仅记录阳性发现",
    "not_applicable": "不适用",
}
FIELD_LABELS = {
    "presence_state": "存在状态",
    "image_mode": "图像模式",
    "annotation_scope": "本图判断范围",
    "subtype": "目标子型",
}


class ReviewQueueEntryWindow(QMainWindow):
    """Record one isolated reviewer slot; never show another reviewer or legacy labels."""

    def __init__(
        self,
        rows: list[Mapping[str, Any]],
        config: Mapping[str, Any],
        reviewer_slot: int,
        repository: ReviewImageRepository,
        save_callback: Callable[[list[dict[str, str]]], None],
    ) -> None:
        super().__init__()
        self._rows = [dict(row) for row in rows]
        self._config = dict(config)
        self._slot = reviewer_slot
        self._prefix = reviewer_prefix(reviewer_slot)
        self._repository = repository
        self._save_callback = save_callback
        self._index = 0
        self._loading_form = False
        self._dirty = False
        self._source_pixmap = QPixmap()
        validate_reviewer_response_rows(
            self._rows, self._config, self._slot, require_complete=False
        )

        self.setWindowTitle(f"区域/病变临床复核 · 复核者{reviewer_slot}")
        self.setStyleSheet(ENTRY_STYLE)
        self.setMinimumSize(900, 640)
        self.resize(1220, 800)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_progress())
        root_layout.addWidget(self._build_content(), 1)
        root_layout.addLayout(self._build_navigation())
        self.setCentralWidget(root)

        self._shortcuts = [
            QShortcut(QKeySequence("Alt+S"), self, activated=self.confirm_current),
            QShortcut(
                QKeySequence("Alt+Left"),
                self,
                activated=lambda: self.navigate(-1),
            ),
            QShortcut(
                QKeySequence("Alt+Right"),
                self,
                activated=lambda: self.navigate(1),
            ),
        ]
        self.show_index(0)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 10, 12, 10)
        title = QLabel("区域/病变临床复核")
        title.setObjectName("title")
        status = QLabel(
            f"正式语义复核 · 独立复核者{self._slot} · 不显示诊断或旧标注"
        )
        status.setObjectName("status")
        status.setAccessibleName("正式盲法独立复核状态")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(status)
        return header

    def _build_progress(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._meta = QLabel()
        self._meta.setObjectName("meta")
        self._meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._progress = QProgressBar()
        self._progress.setRange(0, len(self._rows))
        self._progress.setAccessibleName("已完成复核数量")
        self._entry_status = QLabel("请选择全部必填项后确认。")
        self._entry_status.setObjectName("entryStatus")
        self._entry_status.setAccessibleName("当前录入状态")
        layout.addWidget(self._meta)
        layout.addWidget(self._progress)
        layout.addWidget(self._entry_status)
        return container

    def _build_content(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._target = QLabel()
        self._target.setObjectName("target")
        self._target.setAccessibleName("当前复核目标")
        self._image = QLabel("正在读取ROI…")
        self._image.setObjectName("image")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(480, 380)
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        helper = QLabel(
            "仅显示人工确认ROI。S1a只复核存在状态、模式、范围和子型；"
            "不显示疾病、旧多边形、文件名或路径，不在本阶段修改几何。"
        )
        helper.setObjectName("meta")
        helper.setWordWrap(True)
        image_layout.addWidget(self._target)
        image_layout.addWidget(self._image, 1)
        image_layout.addWidget(helper)

        form_panel = QFrame()
        form_panel.setObjectName("formPanel")
        form_layout = QVBoxLayout(form_panel)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setSpacing(12)
        form_title = QLabel("本图复核结论")
        form_title.setObjectName("sectionTitle")
        form_layout.addWidget(form_title)
        fields = QFormLayout()
        fields.setHorizontalSpacing(12)
        fields.setVerticalSpacing(12)
        self._combos: dict[str, QComboBox] = {}
        for field in REQUIRED_SEMANTIC_FIELDS:
            combo = QComboBox()
            combo.setAccessibleName(FIELD_LABELS[field])
            combo.currentIndexChanged.connect(self._mark_dirty)
            self._combos[field] = combo
            fields.addRow(f"{FIELD_LABELS[field]}：", combo)
        self._combos["presence_state"].currentIndexChanged.connect(
            self._refresh_subtype_for_presence
        )
        form_layout.addLayout(fields)
        notes_label = QLabel("备注（不得填写姓名或路径）：")
        self._notes = QPlainTextEdit()
        self._notes.setAccessibleName("复核备注")
        self._notes.setPlaceholderText("仅记录必要的影像判断说明；可留空。")
        self._notes.setMaximumBlockCount(20)
        self._notes.textChanged.connect(self._mark_dirty)
        form_layout.addWidget(notes_label)
        form_layout.addWidget(self._notes, 1)
        self._confirm = QPushButton("确认并下一张　Alt+S")
        self._confirm.setAccessibleName("确认当前复核并进入下一张")
        self._confirm.clicked.connect(self.confirm_current)
        form_layout.addWidget(self._confirm)

        splitter.addWidget(image_panel)
        splitter.addWidget(form_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([720, 440])
        return splitter

    def _build_navigation(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._previous = QPushButton("← 上一张")
        self._next = QPushButton("下一张 →")
        self._previous.setAccessibleName("上一张复核图像")
        self._next.setAccessibleName("下一张复核图像")
        self._previous.clicked.connect(lambda: self.navigate(-1))
        self._next.clicked.connect(lambda: self.navigate(1))
        layout.addWidget(self._previous)
        layout.addStretch(1)
        layout.addWidget(self._next)
        return layout

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def rows(self) -> list[dict[str, str]]:
        return [dict(row) for row in self._rows]

    def _mark_dirty(self, *_args) -> None:
        if not self._loading_form:
            self._dirty = True
            self._entry_status.setText("当前选择尚未保存。")

    def _set_options(self, combo: QComboBox, values: list[str]) -> None:
        combo.clear()
        combo.addItem("— 请选择 —", "")
        for value in values:
            combo.addItem(f"{VALUE_LABELS.get(value, value)}  [{value}]", value)

    def _select_code(self, combo: QComboBox, code: str) -> None:
        index = combo.findData(code)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _subtype_values_for_presence(self, target: str, presence: str) -> list[str]:
        values = list(self._config["review_fields"]["subtype_by_target"][target])
        if presence in {"absent_visible", "not_in_view"}:
            return ["not_applicable"]
        if presence == "uncertain":
            return [value for value in values if value in {"uncertain", "not_applicable"}]
        if presence == "present":
            return [value for value in values if value != "not_applicable"]
        return values

    def _refresh_subtype_for_presence(self, *_args) -> None:
        if "subtype" not in self._combos or not self._rows:
            return
        target = str(self._rows[self._index]["target_category"])
        presence = str(self._combos["presence_state"].currentData() or "")
        subtype_combo = self._combos["subtype"]
        previous = str(subtype_combo.currentData() or "")
        was_loading = self._loading_form
        self._loading_form = True
        try:
            values = self._subtype_values_for_presence(target, presence)
            self._set_options(subtype_combo, values)
            if previous in values:
                self._select_code(subtype_combo, previous)
            elif len(values) == 1:
                self._select_code(subtype_combo, values[0])
        finally:
            self._loading_form = was_loading
        subtype_combo.setEnabled(
            self._confirm.isEnabled()
            and presence not in {"absent_visible", "not_in_view"}
        )
        if not was_loading:
            self._mark_dirty()

    def _load_form(self, row: Mapping[str, Any]) -> None:
        self._loading_form = True
        try:
            fields = self._config["review_fields"]
            target = str(row["target_category"])
            for field in ("presence_state", "image_mode", "annotation_scope"):
                self._set_options(self._combos[field], list(fields[field]))
            self._set_options(
                self._combos["subtype"], list(fields["subtype_by_target"][target])
            )
            for field in ("presence_state", "image_mode", "annotation_scope"):
                self._select_code(
                    self._combos[field], str(row[f"{self._prefix}_{field}"])
                )
            self._refresh_subtype_for_presence()
            self._select_code(
                self._combos["subtype"], str(row[f"{self._prefix}_subtype"])
            )
            self._notes.setPlainText(str(row[f"{self._prefix}_notes"]))
            self._dirty = False
        finally:
            self._loading_form = False

    def _update_row_from_form(self) -> None:
        row = self._rows[self._index]
        for field in REQUIRED_SEMANTIC_FIELDS:
            row[f"{self._prefix}_{field}"] = str(
                self._combos[field].currentData() or ""
            )
        row[f"{self._prefix}_polygon_action"] = "not_applicable"
        row[f"{self._prefix}_notes"] = self._notes.toPlainText().strip()

    def _set_entry_enabled(self, enabled: bool) -> None:
        for combo in self._combos.values():
            combo.setEnabled(enabled)
        presence = str(self._combos["presence_state"].currentData() or "")
        if presence in {"absent_visible", "not_in_view"}:
            self._combos["subtype"].setEnabled(False)
        self._notes.setEnabled(enabled)
        self._confirm.setEnabled(enabled)

    def _persist_current(self, *, require_current_complete: bool) -> bool:
        self._update_row_from_form()
        missing = [
            field
            for field in REQUIRED_SEMANTIC_FIELDS
            if not self._rows[self._index][f"{self._prefix}_{field}"]
        ]
        if require_current_complete and missing:
            self._entry_status.setText(
                "尚未确认：请填写" + "、".join(FIELD_LABELS[field] for field in missing)
            )
            self._combos[missing[0]].setFocus()
            return False
        try:
            result = validate_reviewer_response_rows(
                self._rows, self._config, self._slot, require_complete=False
            )
            self._save_callback(self.rows)
        except Exception as error:
            self._entry_status.setText(f"保存失败：{error}")
            return False
        self._dirty = False
        self._progress.setValue(int(result["completed_rows"]))
        self._entry_status.setText(
            f"已安全保存；完成 {result['completed_rows']} / {result['rows']}。"
        )
        return True

    def confirm_current(self) -> None:
        if not self._persist_current(require_current_complete=True):
            return
        if self._index < len(self._rows) - 1:
            self.show_index(self._index + 1)

    def navigate(self, delta: int) -> None:
        target = max(0, min(len(self._rows) - 1, self._index + delta))
        if target == self._index:
            return
        if self._dirty and not self._persist_current(require_current_complete=False):
            return
        self.show_index(target)

    def show_index(self, index: int) -> None:
        self._index = index
        row = self._rows[index]
        case_key = str(row["review_case_key"])
        target = str(row["target_category"])
        self._target.setText(f"复核目标：{target}")
        self._meta.setText(f"{index + 1} / {len(self._rows)}　病例：{case_key}")
        self._previous.setEnabled(index > 0)
        self._next.setEnabled(index < len(self._rows) - 1)
        self._confirm.setText(
            "确认并保存　Alt+S"
            if index == len(self._rows) - 1
            else "确认并下一张　Alt+S"
        )
        self._load_form(row)
        try:
            image = self._repository.load_roi(str(row["image_key"]))
            self._source_pixmap = QPixmap.fromImage(image)
            self._image.setAccessibleName(f"{case_key}的ROI超声图像，目标{target}")
            self._set_entry_enabled(True)
            self._render_pixmap()
        except Exception as error:
            self._source_pixmap = QPixmap()
            self._image.setPixmap(QPixmap())
            self._image.setText(f"无法显示当前ROI：{error}")
            self._image.setAccessibleName("当前ROI读取失败")
            self._set_entry_enabled(False)
        result = validate_reviewer_response_rows(
            self._rows, self._config, self._slot, require_complete=False
        )
        self._progress.setValue(int(result["completed_rows"]))
        self._entry_status.setText(
            "当前ROI读取失败，禁止录入。"
            if self._source_pixmap.isNull()
            else "请选择全部必填项后确认。"
        )

    def _render_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        self._image.setPixmap(
            self._source_pixmap.scaled(
                self._image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render_pixmap()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._dirty and not self._persist_current(require_current_complete=False):
            event.ignore()
            return
        event.accept()
