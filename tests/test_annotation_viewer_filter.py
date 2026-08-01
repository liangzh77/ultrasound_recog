import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

import annotation_viewer as viewer_module
from annotation_viewer import FileTree, ImageViewer, MainWindow


class CropFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _build_tree(self, root: Path) -> tuple[FileTree, object]:
        patient = root / "脊柱关节炎" / "SPA1"
        patient.mkdir(parents=True)

        annotations = {
            "a_reviewed": {
                "objects": [],
                "ultrasound_rect": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "ultrasound_rect_reviewed": True,
            },
            "b_automatic": {
                "objects": [],
                "ultrasound_rect": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "ultrasound_rect_reviewed": False,
            },
            "c_not_detected": {"objects": []},
        }
        for stem, annotation in annotations.items():
            patient.joinpath(f"{stem}.jpg").write_bytes(b"image")
            patient.joinpath(f"{stem}.json").write_text(
                json.dumps(annotation),
                encoding="utf-8",
            )

        tree = FileTree()
        tree.populate(root)
        disease_item = tree.topLevelItem(0)
        tree.expandItem(disease_item)
        self.app.processEvents()
        return tree, disease_item.child(0)

    @staticmethod
    def _hidden_by_name(patient_item) -> dict[str, bool]:
        return {
            patient_item.child(index).text(0): patient_item.child(index).isHidden()
            for index in range(patient_item.childCount())
        }

    def test_uncropped_filter_only_shows_images_without_a_rect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tree, patient_item = self._build_tree(Path(temp_dir))
            tree.set_filter_mode(FileTree.FILTER_UNCROPPED)
            hidden_by_name = self._hidden_by_name(patient_item)

            self.assertTrue(hidden_by_name["a_reviewed.jpg"])
            self.assertTrue(hidden_by_name["b_automatic.jpg"])
            self.assertFalse(hidden_by_name["c_not_detected.jpg"])

    def test_unreviewed_filter_only_shows_detected_but_unconfirmed_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tree, patient_item = self._build_tree(Path(temp_dir))
            tree.set_filter_mode(FileTree.FILTER_UNREVIEWED)
            hidden_by_name = self._hidden_by_name(patient_item)

            self.assertTrue(hidden_by_name["a_reviewed.jpg"])
            self.assertFalse(hidden_by_name["b_automatic.jpg"])
            self.assertTrue(hidden_by_name["c_not_detected.jpg"])
            first_visible = MainWindow._first_image_child(None, patient_item)
            self.assertEqual(first_visible.text(0), "b_automatic.jpg")

    def test_opening_an_automatic_crop_does_not_mark_it_reviewed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "automatic.png"
            image = QImage(20, 20, QImage.Format.Format_RGB32)
            image.fill(QColor("black"))
            self.assertTrue(image.save(str(image_path)))
            json_path = image_path.with_suffix(".json")
            json_path.write_text(
                json.dumps(
                    {
                        "objects": [],
                        "ultrasound_rect": {
                            "x1": 0,
                            "y1": 0,
                            "x2": 10,
                            "y2": 10,
                        },
                        "ultrasound_rect_reviewed": False,
                    }
                ),
                encoding="utf-8",
            )

            viewer = ImageViewer()
            viewer.load_image(image_path, {})

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(data["ultrasound_rect_reviewed"])


class CrossDiseaseNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_sample(patient: Path, stem: str, reviewed: bool):
        patient.mkdir(parents=True, exist_ok=True)
        image_path = patient / f"{stem}.png"
        image = QImage(20, 20, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        if not image.save(str(image_path)):
            raise RuntimeError(f"Could not create test image: {image_path}")
        image_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "objects": [],
                    "ultrasound_rect": {
                        "x1": 1,
                        "y1": 1,
                        "x2": 19,
                        "y2": 19,
                        "width": 18,
                        "height": 18,
                    },
                    "ultrasound_rect_reviewed": reviewed,
                }
            ),
            encoding="utf-8",
        )
        return image_path

    def _create_window(self, root: Path) -> MainWindow:
        with patch.object(viewer_module, "DATA_ROOT", root):
            window = MainWindow({})
        window._btn_unreviewed.setChecked(True)
        window._apply_tree_filter()
        return window

    def test_crosses_collapsed_diseases_and_skips_one_without_pending_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "A类" / "patient", "1", reviewed=False)
            self._write_sample(root / "B类" / "patient", "1", reviewed=True)
            expected = self._write_sample(
                root / "C类" / "patient",
                "1",
                reviewed=False,
            )
            window = self._create_window(root)
            try:
                first_disease = window._tree.topLevelItem(0)
                first_disease.setExpanded(True)
                self.app.processEvents()
                start_item = first_disease.child(0).child(0)
                window._tree.setCurrentItem(start_item)
                self.app.processEvents()

                window._on_crop_btn()
                window._on_crop_btn()
                self.app.processEvents()

                next_item = window._tree.currentItem()
                self.assertIsNotNone(next_item)
                self.assertEqual(
                    next_item.data(0, viewer_module.Qt.ItemDataRole.UserRole),
                    expected,
                )
            finally:
                window.close()

    def test_confirming_first_image_in_next_disease_advances_within_that_disease(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root / "A类" / "patient", "1", reviewed=True)
            self._write_sample(root / "B类" / "patient", "1", reviewed=False)
            expected = self._write_sample(
                root / "B类" / "patient",
                "2",
                reviewed=False,
            )
            window = self._create_window(root)
            try:
                second_disease = window._tree.topLevelItem(1)
                second_disease.setExpanded(True)
                self.app.processEvents()
                first_item = second_disease.child(0).child(0)
                window._tree.setCurrentItem(first_item)
                self.app.processEvents()

                window._on_crop_btn()
                window._on_crop_btn()
                self.app.processEvents()

                current_item = window._tree.currentItem()
                self.assertIsNotNone(current_item)
                self.assertEqual(
                    current_item.data(0, viewer_module.Qt.ItemDataRole.UserRole),
                    expected,
                )
            finally:
                window.close()

    def test_viewing_snapshot_item_marks_reviewed_and_keeps_it_visible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = self._write_sample(
                root / "A类" / "patient",
                "1",
                reviewed=False,
            )
            self._write_sample(
                root / "A类" / "patient",
                "2",
                reviewed=False,
            )
            window = self._create_window(root)
            try:
                disease = window._tree.topLevelItem(0)
                disease.setExpanded(True)
                self.app.processEvents()
                patient = disease.child(0)
                first_item = patient.child(0)
                second_item = patient.child(1)
                unreviewed_color = second_item.foreground(0).color().name()

                window._tree.setCurrentItem(first_item)
                self.app.processEvents()

                data = json.loads(
                    first_path.with_suffix(".json").read_text(encoding="utf-8")
                )
                self.assertTrue(data["ultrasound_rect_reviewed"])
                self.assertFalse(first_item.isHidden())
                self.assertNotEqual(
                    first_item.foreground(0).color().name(),
                    unreviewed_color,
                )
                self.assertIn("已确认", first_item.toolTip(0))

                window._on_crop_btn()
                self.app.processEvents()
                self.assertFalse(first_item.isHidden())

                window._apply_tree_filter()
                self.assertFalse(first_item.isHidden())
            finally:
                window.close()

    def test_reactivating_unreviewed_filter_takes_a_new_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(
                root / "A类" / "patient",
                "1",
                reviewed=False,
            )
            self._write_sample(
                root / "A类" / "patient",
                "2",
                reviewed=False,
            )
            window = self._create_window(root)
            try:
                disease = window._tree.topLevelItem(0)
                disease.setExpanded(True)
                self.app.processEvents()
                patient = disease.child(0)
                first_item = patient.child(0)
                second_item = patient.child(1)
                window._tree.setCurrentItem(first_item)
                self.app.processEvents()

                window._btn_unreviewed.click()
                window._btn_unreviewed.click()
                self.app.processEvents()

                self.assertTrue(first_item.isHidden())
                self.assertFalse(second_item.isHidden())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
