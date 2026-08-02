import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel

from src.research_annotation_adjudication_ui import ReviewAdjudicationWindow
from src.research_annotation_review import PUBLIC_REVIEW_FIELDS
from src.research_annotation_review_entry import (
    merge_independent_reviewer_rows,
    new_adjudication_rows,
    new_reviewer_response_rows,
)
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
            "review_workflow_git_commit": "d" * 40,
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
            "presence_state": ["present", "absent_visible", "not_in_view", "uncertain"],
            "image_mode": ["B", "PD", "CD", "unknown"],
            "annotation_scope": [
                "exhaustive_for_declared_targets",
                "positive_only",
                "uncertain",
            ],
            "polygon_action": ["keep", "adjust", "add", "remove", "not_applicable"],
            "subtype_by_target": {
                "积液": ["joint_effusion", "other_fluid", "uncertain", "not_applicable"]
            },
        },
    }


def _source(root: Path):
    image_path = root / "source.png"
    image = QImage(100, 80, QImage.Format.Format_RGB32)
    image.fill(0xFF203040)
    assert image.save(str(image_path))
    annotation_path = root / "source.json"
    annotation_path.write_text(
        json.dumps(
            {
                "ultrasound_rect": {"x1": 10, "y1": 10, "x2": 90, "y2": 70},
                "objects": [{"category": "旧答案"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return image_path, annotation_path


def _merged(disagree=True):
    queue = [{field: "" for field in PUBLIC_REVIEW_FIELDS}]
    queue[0].update(
        {
            "review_case_key": "KNEE_REVIEW_TEST",
            "image_key": "KNEE_IMG_TEST",
            "target_category": "积液",
            "required_independent_reviews": "2",
        }
    )
    responses = []
    for slot in (1, 2):
        rows = new_reviewer_response_rows(queue, _config(), slot)
        prefix = f"reviewer_{slot}"
        rows[0].update(
            {
                f"{prefix}_presence_state": (
                    "absent_visible" if disagree and slot == 2 else "present"
                ),
                f"{prefix}_image_mode": "B",
                f"{prefix}_annotation_scope": "exhaustive_for_declared_targets",
                f"{prefix}_subtype": (
                    "not_applicable" if disagree and slot == 2 else "joint_effusion"
                ),
            }
        )
        responses.append(rows)
    return new_adjudication_rows(
        merge_independent_reviewer_rows(responses[0], responses[1], _config()),
        _config(),
    )


def _window(tmp_path, rows):
    app = QApplication.instance() or QApplication([])
    image_path, annotation_path = _source(tmp_path)
    repository = ReviewImageRepository(
        tmp_path,
        [
            {
                "image_key": "KNEE_IMG_TEST",
                "raw_image_path": image_path.name,
                "normalized_annotation_path": annotation_path.name,
            }
        ],
    )
    saved = []
    window = ReviewAdjudicationWindow(
        rows, _config(), repository, lambda value: saved.append(value)
    )
    app.processEvents()
    return app, window, saved


def test_adjudication_window_only_enables_disagreement_and_saves(tmp_path):
    app, window, saved = _window(tmp_path, _merged(disagree=True))
    try:
        visible = " ".join(label.text() for label in window.findChildren(QLabel))
        assert "KNEE_REVIEW_TEST" in visible
        assert "明确存在" in visible
        assert "可见范围内明确不存在" in visible
        assert "旧答案" not in visible
        assert window._combos["presence_state"].isEnabled()
        assert window._combos["subtype"].isEnabled()
        assert not window._combos["image_mode"].isEnabled()
        assert all(combo.accessibleName() for combo in window._combos.values())
        index = window._combos["presence_state"].findData("present")
        window._combos["presence_state"].setCurrentIndex(index)
        subtype_index = window._combos["subtype"].findData("joint_effusion")
        window._combos["subtype"].setCurrentIndex(subtype_index)
        window._notes.setPlainText("consensus based on visible fluid")
        window.confirm_current()
        app.processEvents()

        assert len(saved) == 1
        assert saved[0][0]["adjudicated_presence_state"] == "present"
        assert saved[0][0]["adjudicated_image_mode"] == ""
        assert window._progress.value() == 1
    finally:
        window.close()


def test_agreed_case_cannot_be_overridden_and_needs_no_notes(tmp_path):
    app, window, saved = _window(tmp_path, _merged(disagree=False))
    try:
        assert all(not combo.isEnabled() for combo in window._combos.values())
        assert not window._notes.isEnabled()
        window.confirm_current()
        app.processEvents()
        assert len(saved) == 1
        assert saved[0][0]["adjudication_notes"] == ""
        assert window._progress.value() == 1
    finally:
        window.close()


def test_presence_adjudication_constrains_subtype_choices(tmp_path):
    app, window, _saved = _window(tmp_path, _merged(disagree=True))
    try:
        presence = window._combos["presence_state"]
        presence.setCurrentIndex(presence.findData("absent_visible"))
        app.processEvents()
        subtype = window._combos["subtype"]
        assert subtype.currentData() == "not_applicable"
        assert not subtype.isEnabled()

        presence.setCurrentIndex(presence.findData("present"))
        app.processEvents()
        assert subtype.findData("not_applicable") < 0
        assert subtype.isEnabled()
    finally:
        window.close()


def test_adjudication_window_rejects_path_notes_and_missing_roi(tmp_path):
    app, window, saved = _window(tmp_path, _merged(disagree=True))
    try:
        index = window._combos["presence_state"].findData("present")
        window._combos["presence_state"].setCurrentIndex(index)
        subtype_index = window._combos["subtype"].findData("joint_effusion")
        window._combos["subtype"].setCurrentIndex(subtype_index)
        window._notes.setPlainText("C:/private/patient/image.png")
        window.confirm_current()
        app.processEvents()
        assert saved == []
        assert "保存失败" in window._status.text()
    finally:
        window._notes.clear()
        window.close()

    missing = ReviewAdjudicationWindow(
        _merged(disagree=True),
        _config(),
        ReviewImageRepository(tmp_path, []),
        lambda _value: None,
    )
    try:
        app.processEvents()
        assert not missing._confirm.isEnabled()
        assert "禁止裁决" in missing._status.text()
    finally:
        missing.close()
