import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from annotation_viewer import ImageViewer


class CropCandidateSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_click_jumps_directly_to_candidate_nearest_the_click(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image = QImage(100, 100, QImage.Format.Format_RGB32)
            image.fill(QColor("black"))
            self.assertTrue(image.save(str(image_path)))

            viewer = ImageViewer()
            viewer.load_image(image_path, {})
            viewer.enter_crop_mode(QRectF(10, 10, 80, 80))
            viewer.set_edge_candidates(
                {
                    "left": [],
                    "right": [55, 65, 75, 85, 90],
                    "top": [],
                    "bottom": [],
                }
            )

            self.assertTrue(viewer._jump_edge_candidate(QPointF(57, 50)))
            self.assertAlmostEqual(viewer.get_crop_rect().right(), 55)


if __name__ == "__main__":
    unittest.main()
