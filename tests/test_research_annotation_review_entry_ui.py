import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel

from src.research_annotation_review import PUBLIC_REVIEW_FIELDS
from src.research_annotation_review_entry import new_reviewer_response_rows
from src.research_annotation_review_entry_ui import ReviewQueueEntryWindow
from src.research_annotation_review_ui import ReviewImageRepository


def _config():
    return {
        "status": "frozen_preregistered",
        "study": "knee_annotation_review_queue_s1a",
        "dataset_fingerprint": "a" * 64,
        "annotation_version": "annotations_aaaaaaaaaaaa",
        "frozen_provenance": {
            "clinical_confirmation_report_sha256": "b" * 64,
            "queue_sha256": "c" * 64,
            "queue_rows": 1,
            "preregistration_git_commit": "d" * 40,
        },
        "required_independent_reviews": 2,
        "selection": {
            "diagnosis_visible_to_reviewer": False,
            "legacy_annotation_visible_to_reviewer": False,
        },
        "review_protocol": {
            "stage": "semantic_presence_only",
            "legacy_polygon_visible_to_reviewer": False,
            "reviewer_polygon_capture": False,
            "polygon_action_policy": "fixed_not_applicable",
            "geometry_reliability_stage": "S1b",
        },
        "review_targets": ["积液"],
        "review_fields": {
            "presence_state": [
                "present",
                "absent_visible",
                "not_in_view",
                "uncertain",
            ],
            "image_mode": ["B", "PD", "CD", "unknown"],
            "annotation_scope": [
                "exhaustive_for_declared_targets",
                "positive_only",
                "uncertain",
            ],
            "polygon_action": ["keep", "adjust", "add", "remove", "not_applicable"],
            "subtype_by_target": {
                "积液": [
                    "joint_effusion",
                    "other_fluid",
                    "uncertain",
                    "not_applicable",
                ]
            },
        },
    }


def _write_source(root: Path, stem: str):
    image_path = root / f"{stem}.png"
    image = QImage(100, 80, QImage.Format.Format_RGB32)
    image.fill(0xFF203040)
    assert image.save(str(image_path))
    annotation_path = root / f"{stem}.json"
    annotation_path.write_text(
        json.dumps(
            {
                "ultrasound_rect": {"x1": 10, "y1": 10, "x2": 90, "y2": 70},
                "objects": [{"category": "旧多边形答案"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return image_path, annotation_path


def _queue_row(index: int):
    row = {field: "" for field in PUBLIC_REVIEW_FIELDS}
    row.update(
        {
            "review_case_key": f"KNEE_REVIEW_{index}",
            "image_key": f"KNEE_IMG_{index}",
            "target_category": "积液",
            "required_independent_reviews": "2",
        }
    )
    return row


def _window(tmp_path, count=1, config=None):
    app = QApplication.instance() or QApplication([])
    sources = []
    queue = []
    for index in range(count):
        image_path, annotation_path = _write_source(tmp_path, f"source-{index}")
        sources.append(
            {
                "image_key": f"KNEE_IMG_{index}",
                "raw_image_path": image_path.name,
                "normalized_annotation_path": annotation_path.name,
            }
        )
        queue.append(_queue_row(index))
    selected_config = config or _config()
    selected_config["frozen_provenance"]["queue_rows"] = count
    rows = new_reviewer_response_rows(queue, selected_config, 1)
    saved = []
    window = ReviewQueueEntryWindow(
        rows,
        selected_config,
        1,
        ReviewImageRepository(tmp_path, sources),
        lambda value: saved.append(value),
    )
    app.processEvents()
    return app, window, saved


def _select(window, field, code):
    combo = window._combos[field]
    index = combo.findData(code)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_entry_window_is_blinded_accessible_and_saves_complete_case(tmp_path):
    app, window, saved = _window(tmp_path)
    try:
        visible_text = " ".join(
            label.text() for label in window.findChildren(QLabel)
        )
        assert "KNEE_REVIEW_0" in visible_text
        assert "积液" in visible_text
        assert "旧多边形答案" not in visible_text
        assert str(tmp_path) not in visible_text
        assert all(combo.accessibleName() for combo in window._combos.values())
        assert len(window._shortcuts) == 3

        _select(window, "presence_state", "present")
        _select(window, "image_mode", "B")
        _select(window, "annotation_scope", "exhaustive_for_declared_targets")
        _select(window, "subtype", "joint_effusion")
        window.confirm_current()
        app.processEvents()

        assert len(saved) == 1
        assert saved[0][0]["reviewer_1_presence_state"] == "present"
        assert saved[0][0]["reviewer_1_polygon_action"] == "not_applicable"
        assert saved[0][0]["reviewer_2_presence_state"] == ""
        assert window._progress.value() == 1
    finally:
        window.close()


def test_navigation_persists_partial_progress_without_counting_complete(tmp_path):
    app, window, saved = _window(tmp_path, count=2)
    try:
        _select(window, "presence_state", "uncertain")
        window.navigate(1)
        app.processEvents()

        assert window.current_index == 1
        assert saved[-1][0]["reviewer_1_presence_state"] == "uncertain"
        assert window._progress.value() == 0
    finally:
        window.close()


def test_entry_window_refuses_forbidden_path_in_notes(tmp_path):
    app, window, saved = _window(tmp_path)
    try:
        for field, code in {
            "presence_state": "present",
            "image_mode": "B",
            "annotation_scope": "exhaustive_for_declared_targets",
            "subtype": "joint_effusion",
        }.items():
            _select(window, field, code)
        window._notes.setPlainText("C:/private/patient/image.png")
        window.confirm_current()
        app.processEvents()

        assert saved == []
        assert "保存失败" in window._entry_status.text()
    finally:
        window._notes.clear()
        window.close()


def test_entry_window_refuses_unfrozen_config(tmp_path):
    config = _config()
    config["status"] = "draft_not_preregistered"
    with pytest.raises(ValueError, match="frozen_preregistered"):
        _window(tmp_path, config=config)


def test_entry_window_disables_form_when_roi_cannot_load(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = _config()
    rows = new_reviewer_response_rows([_queue_row(0)], config, 1)
    window = ReviewQueueEntryWindow(
        rows,
        config,
        1,
        ReviewImageRepository(tmp_path, []),
        lambda _value: None,
    )
    try:
        app.processEvents()
        assert window._source_pixmap.isNull()
        assert not window._confirm.isEnabled()
        assert all(not combo.isEnabled() for combo in window._combos.values())
        assert "禁止录入" in window._entry_status.text()
    finally:
        window.close()
