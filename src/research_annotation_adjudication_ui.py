"""Diagnosis-blinded S1a disagreement adjudication window."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
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
    validate_adjudication_rows,
)
from src.research_annotation_review_entry_ui import (
    ENTRY_STYLE,
    FIELD_LABELS,
    VALUE_LABELS,
)
from src.research_annotation_review_ui import ReviewImageRepository


class ReviewAdjudicationWindow(QMainWindow):
    """Resolve only semantic disagreements without diagnosis or legacy overlays."""

    def __init__(
        self,
        rows: list[Mapping[str, Any]],
        config: Mapping[str, Any],
        repository: ReviewImageRepository,
        save_callback: Callable[[list[dict[str, str]]], None],
    ) -> None:
        super().__init__()
        self._rows = [dict(row) for row in rows]
        self._config = dict(config)
        self._repository = repository
        self._save_callback = save_callback
        self._index = 0
        self._loading = False
        self._dirty = False
        self._source_pixmap = QPixmap()
        validate_adjudication_rows(self._rows, self._config, require_complete=False)

        self.setWindowTitle("区域/病变临床复核 · 盲法分歧裁决")
        self.setStyleSheet(ENTRY_STYLE)
        self.setMinimumSize(940, 660)
        self.resize(1260, 820)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_progress())
        layout.addWidget(self._build_content(), 1)
        layout.addLayout(self._build_navigation())
        self.setCentralWidget(root)
        self._shortcuts = [
            QShortcut(QKeySequence("Alt+S"), self, activated=self.confirm_current),
            QShortcut(
                QKeySequence("Alt+Left"), self, activated=lambda: self.navigate(-1)
            ),
            QShortcut(
                QKeySequence("Alt+Right"), self, activated=lambda: self.navigate(1)
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
        status = QLabel("盲法分歧裁决 · 仅能修改分歧字段 · 不显示诊断或旧标注")
        status.setObjectName("status")
        status.setAccessibleName("盲法分歧裁决状态")
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
        self._progress = QProgressBar()
        self._progress.setRange(0, len(self._rows))
        self._progress.setAccessibleName("已完成裁决数量")
        self._status = QLabel("仅对双方分歧字段作出裁决。")
        self._status.setObjectName("entryStatus")
        self._status.setAccessibleName("当前裁决状态")
        layout.addWidget(self._meta)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        return container

    def _build_content(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._target = QLabel()
        self._target.setObjectName("target")
        self._image = QLabel("正在读取ROI…")
        self._image.setObjectName("image")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(480, 380)
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        helper = QLabel(
            "仅显示人工确认ROI与两位复核者的代码结论。"
            "不显示诊断、旧标注、身份、文件名或路径。"
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
        heading = QLabel("双方结论与裁决")
        heading.setObjectName("sectionTitle")
        form_layout.addWidget(heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for column, text in enumerate(("字段", "复核者1", "复核者2", "裁决")):
            label = QLabel(text)
            label.setObjectName("meta")
            grid.addWidget(label, 0, column)
        self._reviewer_labels: dict[tuple[int, str], QLabel] = {}
        self._combos: dict[str, QComboBox] = {}
        for row_index, field in enumerate(REQUIRED_SEMANTIC_FIELDS, start=1):
            grid.addWidget(QLabel(FIELD_LABELS[field]), row_index, 0)
            for slot in (1, 2):
                label = QLabel()
                label.setWordWrap(True)
                label.setAccessibleName(f"复核者{slot}{FIELD_LABELS[field]}")
                self._reviewer_labels[(slot, field)] = label
                grid.addWidget(label, row_index, slot)
            combo = QComboBox()
            combo.setAccessibleName(f"裁决{FIELD_LABELS[field]}")
            combo.currentIndexChanged.connect(self._mark_dirty)
            self._combos[field] = combo
            grid.addWidget(combo, row_index, 3)
        self._combos["presence_state"].currentIndexChanged.connect(
            self._refresh_subtype_for_presence
        )
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 2)
        form_layout.addLayout(grid)
        notes_label = QLabel("裁决理由（有分歧时必填，不得填写姓名或路径）：")
        self._notes = QPlainTextEdit()
        self._notes.setAccessibleName("裁决理由")
        self._notes.setPlaceholderText("说明选择依据；无分歧病例不可填写。")
        self._notes.setMaximumBlockCount(20)
        self._notes.textChanged.connect(self._mark_dirty)
        form_layout.addWidget(notes_label)
        form_layout.addWidget(self._notes, 1)
        self._confirm = QPushButton("确认裁决并下一张　Alt+S")
        self._confirm.setAccessibleName("确认当前裁决并进入下一张")
        self._confirm.clicked.connect(self.confirm_current)
        form_layout.addWidget(self._confirm)

        splitter.addWidget(image_panel)
        splitter.addWidget(form_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([620, 580])
        return splitter

    def _build_navigation(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._previous = QPushButton("← 上一张")
        self._next = QPushButton("下一张 →")
        self._previous.setAccessibleName("上一张裁决图像")
        self._next.setAccessibleName("下一张裁决图像")
        self._previous.clicked.connect(lambda: self.navigate(-1))
        self._next.clicked.connect(lambda: self.navigate(1))
        layout.addWidget(self._previous)
        layout.addStretch(1)
        layout.addWidget(self._next)
        return layout

    @property
    def rows(self) -> list[dict[str, str]]:
        return [dict(row) for row in self._rows]

    def _mark_dirty(self, *_args) -> None:
        if not self._loading:
            self._dirty = True
            self._status.setText("当前裁决尚未保存。")

    def _allowed_for(self, target: str, field: str) -> list[str]:
        fields = self._config["review_fields"]
        return (
            list(fields["subtype_by_target"][target])
            if field == "subtype"
            else list(fields[field])
        )

    def _subtype_values_for_presence(self, target: str, presence: str) -> list[str]:
        values = self._allowed_for(target, "subtype")
        if presence in {"absent_visible", "not_in_view"}:
            return ["not_applicable"]
        if presence == "uncertain":
            return [value for value in values if value in {"uncertain", "not_applicable"}]
        if presence == "present":
            return [value for value in values if value != "not_applicable"]
        return values

    def _resolved_presence(self, row: Mapping[str, Any]) -> str:
        if str(row["reviewer_1_presence_state"]) == str(
            row["reviewer_2_presence_state"]
        ):
            return str(row["reviewer_1_presence_state"])
        return str(self._combos["presence_state"].currentData() or "")

    def _refresh_subtype_for_presence(self, *_args) -> None:
        if "subtype" not in self._combos or not self._rows:
            return
        row = self._rows[self._index]
        if str(row["reviewer_1_subtype"]) == str(row["reviewer_2_subtype"]):
            return
        combo = self._combos["subtype"]
        previous = str(combo.currentData() or row["adjudicated_subtype"] or "")
        target = str(row["target_category"])
        values = self._subtype_values_for_presence(
            target, self._resolved_presence(row)
        )
        was_loading = self._loading
        self._loading = True
        try:
            combo.clear()
            combo.addItem("— 请选择裁决 —", "")
            for value in values:
                combo.addItem(f"{VALUE_LABELS.get(value, value)}  [{value}]", value)
            if previous in values:
                combo.setCurrentIndex(combo.findData(previous))
            elif len(values) == 1:
                combo.setCurrentIndex(1)
        finally:
            self._loading = was_loading
        combo.setEnabled(self._confirm.isEnabled() and len(values) > 1)
        if not was_loading:
            self._mark_dirty()

    def _load_form(self, row: Mapping[str, Any]) -> list[str]:
        self._loading = True
        disagreements = []
        try:
            target = str(row["target_category"])
            for field in REQUIRED_SEMANTIC_FIELDS:
                left = str(row[f"reviewer_1_{field}"])
                right = str(row[f"reviewer_2_{field}"])
                self._reviewer_labels[(1, field)].setText(
                    VALUE_LABELS.get(left, left)
                )
                self._reviewer_labels[(2, field)].setText(
                    VALUE_LABELS.get(right, right)
                )
                combo = self._combos[field]
                combo.clear()
                if left == right:
                    combo.addItem("双方一致，无需裁决", "")
                    combo.setEnabled(False)
                else:
                    disagreements.append(field)
                    combo.addItem("— 请选择裁决 —", "")
                    for value in self._allowed_for(target, field):
                        combo.addItem(
                            f"{VALUE_LABELS.get(value, value)}  [{value}]", value
                        )
                    combo.setEnabled(True)
                    selected = combo.findData(str(row[f"adjudicated_{field}"]))
                    combo.setCurrentIndex(selected if selected >= 0 else 0)
            self._refresh_subtype_for_presence()
            self._notes.setPlainText(str(row["adjudication_notes"]))
            self._notes.setEnabled(bool(disagreements))
            self._dirty = False
        finally:
            self._loading = False
        return disagreements

    def _set_entry_enabled(self, enabled: bool) -> None:
        row = self._rows[self._index]
        for field, combo in self._combos.items():
            disagrees = str(row[f"reviewer_1_{field}"]) != str(
                row[f"reviewer_2_{field}"]
            )
            combo.setEnabled(enabled and disagrees)
        if enabled:
            self._refresh_subtype_for_presence()
        has_disagreement = any(
            str(row[f"reviewer_1_{field}"]) != str(row[f"reviewer_2_{field}"])
            for field in REQUIRED_SEMANTIC_FIELDS
        )
        self._notes.setEnabled(enabled and has_disagreement)
        self._confirm.setEnabled(enabled)

    def _update_row(self) -> list[str]:
        row = self._rows[self._index]
        disagreements = []
        for field in REQUIRED_SEMANTIC_FIELDS:
            disagrees = str(row[f"reviewer_1_{field}"]) != str(
                row[f"reviewer_2_{field}"]
            )
            if disagrees:
                disagreements.append(field)
                row[f"adjudicated_{field}"] = str(
                    self._combos[field].currentData() or ""
                )
            else:
                row[f"adjudicated_{field}"] = ""
        row["adjudicated_polygon_action"] = ""
        row["adjudication_notes"] = self._notes.toPlainText().strip()
        return disagreements

    def _persist_current(self, *, require_current_complete: bool) -> bool:
        disagreements = self._update_row()
        row = self._rows[self._index]
        missing = [field for field in disagreements if not row[f"adjudicated_{field}"]]
        if require_current_complete and (missing or (disagreements and not row["adjudication_notes"])):
            labels = [FIELD_LABELS[field] for field in missing]
            if disagreements and not row["adjudication_notes"]:
                labels.append("裁决理由")
            self._status.setText("尚未完成：请填写" + "、".join(labels))
            return False
        try:
            progress = validate_adjudication_rows(
                self._rows, self._config, require_complete=False
            )
            self._save_callback(self.rows)
        except Exception as error:
            self._status.setText(f"保存失败：{error}")
            return False
        self._dirty = False
        self._progress.setValue(int(progress["completed_rows"]))
        self._status.setText(
            f"已安全保存；完成 {progress['completed_rows']} / {progress['rows']}。"
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
        self._target.setText(f"裁决目标：{target}")
        disagreements = self._load_form(row)
        self._meta.setText(
            f"{index + 1} / {len(self._rows)}　病例：{case_key}　"
            f"分歧字段：{len(disagreements)}"
        )
        self._previous.setEnabled(index > 0)
        self._next.setEnabled(index < len(self._rows) - 1)
        self._confirm.setText(
            "确认裁决并保存　Alt+S"
            if index == len(self._rows) - 1
            else "确认裁决并下一张　Alt+S"
        )
        try:
            image = self._repository.load_roi(str(row["image_key"]))
            self._source_pixmap = QPixmap.fromImage(image)
            self._image.setAccessibleName(f"{case_key}的ROI超声图像，裁决目标{target}")
            self._set_entry_enabled(True)
            self._render_pixmap()
        except Exception as error:
            self._source_pixmap = QPixmap()
            self._image.setPixmap(QPixmap())
            self._image.setText(f"无法显示当前ROI：{error}")
            self._image.setAccessibleName("当前裁决ROI读取失败")
            self._set_entry_enabled(False)
        progress = validate_adjudication_rows(
            self._rows, self._config, require_complete=False
        )
        self._progress.setValue(int(progress["completed_rows"]))
        if self._source_pixmap.isNull():
            self._status.setText("当前ROI读取失败，禁止裁决。")
        elif disagreements:
            self._status.setText("仅对分歧字段选择最终代码，并填写裁决理由。")
        else:
            self._status.setText("双方全部一致；确认后记录本例无需裁决。")

    def _render_pixmap(self) -> None:
        if not self._source_pixmap.isNull():
            self._image.setPixmap(
                self._source_pixmap.scaled(
                    self._image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_pixmap()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._dirty and not self._persist_current(require_current_complete=False):
            event.ignore()
            return
        event.accept()
