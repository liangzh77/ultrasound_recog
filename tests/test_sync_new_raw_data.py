import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.sync_new_raw_data import (
    build_sync_plan,
    execute_sync_plan,
    merge_annotation_data,
)


def write_patient(
    disease_dir: Path,
    folder_name: str,
    images: dict[str, bytes],
    *,
    crop: bool = False,
) -> Path:
    patient_dir = disease_dir / folder_name
    patient_dir.mkdir(parents=True, exist_ok=True)
    for image_name, content in images.items():
        image_path = patient_dir / image_name
        image_path.write_bytes(content)
        annotation = {
            "info": {"description": "ISAT", "name": image_name},
            "objects": [{"category": "test", "segmentation": [[0, 0], [1, 0], [1, 1]]}],
        }
        if crop:
            annotation["ultrasound_rect"] = {
                "x1": 1,
                "y1": 2,
                "x2": 11,
                "y2": 12,
                "width": 10,
                "height": 10,
            }
            annotation["ultrasound_rect_reviewed"] = True
        image_path.with_suffix(".json").write_text(
            json.dumps(annotation, ensure_ascii=False),
            encoding="utf-8",
        )
    (patient_dir / "isat.yaml").write_text("label: []\n", encoding="utf-8")
    return patient_dir


class MergeAnnotationDataTests(unittest.TestCase):
    def test_source_annotation_wins_while_target_crop_metadata_is_preserved(self):
        source = {
            "info": {"description": "ISAT", "name": "a.jpg"},
            "objects": [{"category": "new-label"}],
        }
        target = {
            "info": {"description": "ISAT", "name": "a.jpg"},
            "objects": [{"category": "old-label"}],
            "ultrasound_rect": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "ultrasound_candidates": {"left": [1, 2]},
            "ultrasound_rect_reviewed": True,
        }

        merged = merge_annotation_data(source, target)

        self.assertEqual(merged["objects"], source["objects"])
        self.assertEqual(merged["ultrasound_rect"], target["ultrasound_rect"])
        self.assertEqual(
            merged["ultrasound_candidates"],
            target["ultrasound_candidates"],
        )
        self.assertTrue(merged["ultrasound_rect_reviewed"])


class SyncPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.source = root / "source"
        self.target = root / "target"
        self.source.mkdir()
        self.target.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matches_by_complete_image_identity_and_excludes_test_dataset(self):
        source_disease = self.source / "正常"
        target_disease = self.target / "正常"
        write_patient(source_disease, "N1", {"same.jpg": b"one", "a.jpg": b"a"})
        write_patient(source_disease, "N2", {"same.jpg": b"two", "b.jpg": b"b"})
        write_patient(
            target_disease,
            "脱敏甲",
            {"same.jpg": b"one", "a.jpg": b"a"},
            crop=True,
        )
        write_patient(
            target_disease,
            "脱敏乙",
            {"same.jpg": b"two", "b.jpg": b"b"},
            crop=True,
        )
        write_patient(
            self.source / "膝关节2026未标注",
            "测试1",
            {"test.jpg": b"test"},
        )

        plan = build_sync_plan(self.source, self.target)

        rename_pairs = {
            (item.target_dir.name, item.source_dir.name)
            for item in plan.rename_actions
        }
        self.assertEqual(rename_pairs, {("脱敏甲", "N1"), ("脱敏乙", "N2")})
        self.assertNotIn(
            "膝关节2026未标注",
            {item.disease for item in plan.patient_actions},
        )
        self.assertFalse(plan.conflicts)

    def test_unmatched_target_is_preserved_and_new_source_is_copied(self):
        source_disease = self.source / "损伤"
        target_disease = self.target / "损伤"
        write_patient(source_disease, "损伤1", {"known.jpg": b"known"})
        write_patient(source_disease, "损伤2", {"new.jpg": b"new"})
        write_patient(
            target_disease,
            "旧名",
            {"known.jpg": b"known"},
            crop=True,
        )
        write_patient(
            target_disease,
            "仅工作区",
            {"legacy.jpg": b"legacy"},
            crop=True,
        )

        plan = build_sync_plan(self.source, self.target)

        self.assertEqual(
            [item.target_dir.name for item in plan.unmatched_target_actions],
            ["仅工作区"],
        )
        self.assertEqual(
            [item.source_dir.name for item in plan.new_patient_actions],
            ["损伤2"],
        )
        self.assertFalse(plan.conflicts)

    def test_patient_reclassified_to_another_disease_is_moved_not_duplicated(self):
        write_patient(
            self.source / "脊柱关节炎",
            "SPA15",
            {"reclassified.jpg": b"same-patient"},
        )
        write_patient(
            self.target / "损伤",
            "旧损伤患者名",
            {"reclassified.jpg": b"same-patient"},
            crop=True,
        )

        plan = build_sync_plan(self.source, self.target)

        self.assertEqual(len(plan.matched_patient_actions), 1)
        self.assertEqual(len(plan.new_patient_actions), 0)
        self.assertEqual(len(plan.unmatched_target_actions), 0)
        action = plan.matched_patient_actions[0]
        self.assertEqual(action.disease, "脊柱关节炎")
        self.assertEqual(action.target_dir.parent.name, "损伤")
        self.assertFalse(plan.conflicts)
        execute_sync_plan(plan, backup_root=Path(self.temp_dir.name) / "backup")
        moved_json = self.target / "脊柱关节炎" / "SPA15" / "reclassified.json"
        self.assertTrue(moved_json.is_file())
        self.assertIn(
            "ultrasound_rect",
            json.loads(moved_json.read_text(encoding="utf-8")),
        )
        self.assertFalse((self.target / "损伤" / "旧损伤患者名").exists())


class ExecuteSyncPlanTests(unittest.TestCase):
    def test_renames_merges_copies_excel_and_preserves_crop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            source_disease = source / "脊柱关节炎"
            target_disease = target / "脊柱关节炎"
            write_patient(source_disease, "SPA1", {"old.jpg": b"old"})
            write_patient(source_disease, "SPA2", {"new.jpg": b"new"})
            write_patient(
                target_disease,
                "脱敏旧名",
                {"old.jpg": b"old"},
                crop=True,
            )
            (source_disease / "SPA1" / "mask").mkdir()
            (source_disease / "SPA1" / "mask" / "old.png").write_bytes(
                b"new-mask"
            )
            (target_disease / "脱敏旧名" / "mask").mkdir()
            (target_disease / "脱敏旧名" / "mask" / "old.png").write_bytes(
                b"old-mask"
            )
            source_disease.joinpath("SPA编号.xlsx").write_bytes(b"xlsx")
            source.joinpath("训练要求.txt").write_text(
                "按患者分组训练",
                encoding="utf-8",
            )
            write_patient(
                source / "膝关节2026未标注",
                "测试1",
                {"test.jpg": b"test"},
            )

            plan = build_sync_plan(source, target)
            report = execute_sync_plan(plan, backup_root=root / "backup")

            self.assertFalse((target_disease / "脱敏旧名").exists())
            self.assertTrue((target_disease / "SPA1").is_dir())
            self.assertTrue((target_disease / "SPA2").is_dir())
            self.assertTrue((target_disease / "SPA编号.xlsx").is_file())
            self.assertEqual(
                (target.parent / "训练要求.txt").read_text(encoding="utf-8"),
                "按患者分组训练",
            )
            self.assertFalse((target / "训练要求.txt").exists())
            self.assertFalse((target / "膝关节2026未标注").exists())
            self.assertEqual(
                (target_disease / "SPA1" / "mask" / "old.png").read_bytes(),
                b"new-mask",
            )
            merged = json.loads(
                (target_disease / "SPA1" / "old.json").read_text(encoding="utf-8")
            )
            self.assertIn("ultrasound_rect", merged)
            self.assertTrue(merged["ultrasound_rect_reviewed"])
            self.assertGreater(report.files_copied, 0)
            self.assertEqual(report.conflicts, 0)


class CliEntryPointTests(unittest.TestCase):
    def test_script_can_be_run_directly_from_repository_root(self):
        repository_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                str(repository_root / "tools" / "sync_new_raw_data.py"),
                "--help",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--apply", result.stdout)


if __name__ == "__main__":
    unittest.main()
