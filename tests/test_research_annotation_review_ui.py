import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel

from src.research_annotation_review_ui import (
    ReviewImageRepository,
    ReviewQueuePreviewWindow,
    audit_review_queue_rois,
    configure_cjk_font,
    load_roi_qimage,
)


def _write_image_and_annotation(root: Path, stem: str):
    image_path = root / f"{stem}.png"
    image = QImage(100, 80, QImage.Format.Format_RGB32)
    image.fill(0xFF203040)
    assert image.save(str(image_path))
    annotation_path = root / f"{stem}.json"
    annotation_path.write_text(
        json.dumps(
            {
                "ultrasound_rect": {"x1": 20, "y1": 10, "x2": 80, "y2": 60},
                "objects": [{"category": "旧答案不应显示"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return image_path, annotation_path


def test_roi_loader_excludes_pixels_outside_the_confirmed_rectangle():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path, annotation_path = _write_image_and_annotation(root, "sample")

        roi = load_roi_qimage(image_path, annotation_path)

        assert roi.width() == 60
        assert roi.height() == 50


def test_cjk_font_configuration_loads_a_font_when_windows_font_is_available():
    app = QApplication.instance() or QApplication([])
    family = configure_cjk_font(app)
    windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if windows_font.is_file():
        assert family


def test_preview_window_shows_only_pseudonymous_case_and_target():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path, annotation_path = _write_image_and_annotation(root, "sample")
        repository = ReviewImageRepository(
            root,
            [
                {
                    "image_key": "KNEE_IMG_TEST",
                    "raw_image_path": image_path.name,
                    "normalized_annotation_path": annotation_path.name,
                }
            ],
        )
        rows = [
            {
                "review_case_key": "KNEE_REVIEW_TEST",
                "image_key": "KNEE_IMG_TEST",
                "target_category": "积液",
            }
        ]
        window = ReviewQueuePreviewWindow(rows, repository)
        try:
            app.processEvents()
            visible_text = " ".join(
                label.text() for label in window.findChildren(QLabel)
            )
            assert "KNEE_REVIEW_TEST" in visible_text
            assert "积液" in visible_text
            assert "旧答案不应显示" not in visible_text
            assert str(image_path) not in visible_text
            assert not window._source_pixmap.isNull()
        finally:
            window.close()


def test_preview_navigation_has_keyboard_equivalent_and_clamped_bounds():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        sources = []
        rows = []
        for index in range(2):
            image_path, annotation_path = _write_image_and_annotation(
                root, f"sample-{index}"
            )
            image_key = f"KNEE_IMG_{index}"
            sources.append(
                {
                    "image_key": image_key,
                    "raw_image_path": image_path.name,
                    "normalized_annotation_path": annotation_path.name,
                }
            )
            rows.append(
                {
                    "review_case_key": f"KNEE_REVIEW_{index}",
                    "image_key": image_key,
                    "target_category": "积液",
                }
            )
        window = ReviewQueuePreviewWindow(rows, ReviewImageRepository(root, sources))
        try:
            window.navigate(1)
            assert window.current_index == 1
            window.navigate(1)
            assert window.current_index == 1
            window.navigate(-1)
            assert window.current_index == 0
        finally:
            window.close()


def test_roi_queue_audit_reports_aggregate_failures_without_source_paths():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path, annotation_path = _write_image_and_annotation(root, "ok")
        repository = ReviewImageRepository(
            root,
            [
                {
                    "image_key": "KNEE_IMG_OK",
                    "raw_image_path": image_path.name,
                    "normalized_annotation_path": annotation_path.name,
                }
            ],
        )
        result = audit_review_queue_rois(
            [
                {"review_case_key": "R_OK", "image_key": "KNEE_IMG_OK"},
                {"review_case_key": "R_MISSING", "image_key": "KNEE_IMG_MISSING"},
            ],
            repository,
        )

        assert result["loaded_rois"] == 1
        assert result["failed_rois"] == 1
        assert result["failed_review_case_keys"] == ["R_MISSING"]
        assert str(root) not in json.dumps(result)
