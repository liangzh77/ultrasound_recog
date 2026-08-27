from pathlib import Path

import pytest
from PIL import Image

from src.research_blind_cohort import build_blind_image_registry


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 24), color=color).save(path)


def test_build_blind_registry_separates_public_and_private_linkage(tmp_path: Path):
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    for patient, color in ((1, (1, 2, 3)), (2, (4, 5, 6))):
        patient_dir = cohort / str(patient)
        patient_dir.mkdir()
        _write_image(patient_dir / "private_name.jpg", color)

    public, private_sources, private_people, summary = build_blind_image_registry(
        cohort,
        secret=b"stable-test-secret",
        expected_patients=2,
        expected_images=2,
        project_root=tmp_path,
    )

    assert len(public) == 2
    assert len({row["image_key"] for row in public}) == 2
    assert all("private_name" not in str(row) for row in public)
    assert all("raw_image_path" in row for row in private_sources)
    assert {row["hospital_patient_number"] for row in private_people} == {1, 2}
    assert summary["reference_diagnoses_read"] is False
    assert summary["public_contains_raw_paths"] is False


def test_build_blind_registry_rejects_missing_patient_directory(tmp_path: Path):
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    (cohort / "1").mkdir()
    _write_image(cohort / "1" / "image.jpg", (1, 2, 3))
    with pytest.raises(ValueError, match="exactly cover"):
        build_blind_image_registry(
            cohort,
            secret=b"stable-test-secret",
            expected_patients=2,
            expected_images=1,
            project_root=tmp_path,
        )

