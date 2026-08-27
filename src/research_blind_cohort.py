"""Privacy-safe registry construction for the label-hidden 2026 cohort."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

from src.research_manifest import pseudonymous_key


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_blind_image_registry(
    cohort_root: Path,
    *,
    secret: bytes,
    expected_patients: int,
    expected_images: int,
    project_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict]:
    """Build public image rows and private linkage without reading any labels."""

    if not secret:
        raise ValueError("pseudonymization secret must not be empty")
    expected_numbers = set(range(1, expected_patients + 1))
    patient_dirs: dict[int, Path] = {}
    for path in cohort_root.iterdir():
        if not path.is_dir():
            continue
        try:
            patient_number = int(path.name)
        except ValueError as error:
            raise ValueError("blind patient directories must use integer identifiers") from error
        if str(patient_number) != path.name or patient_number in patient_dirs:
            raise ValueError("blind patient directory identifiers must be unique canonical integers")
        patient_dirs[patient_number] = path
    if set(patient_dirs) != expected_numbers:
        raise ValueError("blind patient directories do not exactly cover the frozen identifiers")

    public_images: list[dict[str, object]] = []
    private_sources: list[dict[str, object]] = []
    private_people: list[dict[str, object]] = []
    content_hash_counts: Counter[str] = Counter()
    images_per_patient: list[int] = []

    for patient_number in sorted(patient_dirs):
        person_key = pseudonymous_key(
            f"2026:{patient_number}", secret, "KNEE_EXT"
        )
        private_people.append(
            {
                "person_key": person_key,
                "hospital_patient_number": patient_number,
            }
        )
        image_paths = sorted(
            (
                path
                for path in patient_dirs[patient_number].rglob("*")
                if path.is_file() and path.suffix.casefold() == ".jpg"
            ),
            key=lambda path: path.relative_to(patient_dirs[patient_number]).as_posix().casefold(),
        )
        if not image_paths:
            raise ValueError("one or more blind patients have no JPG images")
        images_per_patient.append(len(image_paths))
        for image_index, image_path in enumerate(image_paths, start=1):
            content_sha256 = _file_sha256(image_path)
            content_hash_counts[content_sha256] += 1
            image_key = pseudonymous_key(
                f"{person_key}:{image_index}:{content_sha256}", secret, "KNEE_EXT_IMG"
            )
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("blind image dimensions must be positive")
            public_images.append(
                {
                    "image_key": image_key,
                    "person_key": person_key,
                    "image_index": image_index,
                    "width": width,
                    "height": height,
                    "include": 1,
                }
            )
            private_sources.append(
                {
                    "image_key": image_key,
                    "person_key": person_key,
                    "raw_image_path": image_path.resolve()
                    .relative_to(project_root.resolve())
                    .as_posix(),
                    "content_sha256": content_sha256,
                }
            )

    if len(public_images) != expected_images:
        raise ValueError(
            f"expected {expected_images} blind images, found {len(public_images)}"
        )
    image_keys = [str(row["image_key"]) for row in public_images]
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("blind image pseudonymous keys are not unique")
    summary = {
        "patients": len(private_people),
        "images": len(public_images),
        "minimum_images_per_patient": min(images_per_patient),
        "maximum_images_per_patient": max(images_per_patient),
        "duplicate_content_hash_groups": sum(
            count > 1 for count in content_hash_counts.values()
        ),
        "public_contains_raw_paths": False,
        "reference_diagnoses_read": False,
    }
    return public_images, private_sources, private_people, summary

