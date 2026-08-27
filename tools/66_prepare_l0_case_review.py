"""Prepare and render the deterministic private L0 lesion-mask case review."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_l0_case_review import build_case_review_queue  # noqa: E402
from src.research_lesion_guided import roi_pixel_bounds  # noqa: E402
from src.research_transforms import letterbox_rgb  # noqa: E402


REPORT_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "lesion_guided"
OOF_DIR = PATIENT_MULTIMODAL_REPORTS_DIR / "oof"
MASK_MANIFEST = REPORT_DIR / "L0_manual_lesion_mask_manifest.csv"
PATIENT_QUEUE = REPORT_DIR / "L0_case_review_queue.csv"
IMAGE_QUEUE = REPORT_DIR / "L0_case_review_images.csv"
SUMMARY_PATH = REPORT_DIR / "L0_case_review_summary.json"
PRIVATE_DIR = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "l0_case_review_visuals"
OUTPUT_SIZE = 384
TILE_SIZE = 192
PAGE_WIDTH = 1600
PATIENTS_PER_PAGE = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_project_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Private review paths must remain project-relative")
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/arial.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render_overlay(image_row: dict[str, str], source_path: Path, mask_path: Path) -> Image.Image:
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    roi = {axis: float(image_row[f"roi_{axis}"]) for axis in ("x1", "y1", "x2", "y2")}
    bounds = roi_pixel_bounds(roi, source.width, source.height)
    rgb = letterbox_rgb(source.crop(bounds), OUTPUT_SIZE, fill=(0, 0, 0))
    with Image.open(mask_path) as opened_mask:
        mask = opened_mask.convert("L")
    if mask.size != rgb.size:
        raise ValueError("L0 review RGB and mask are not co-registered")
    overlay = Image.new("RGB", rgb.size, (255, 32, 32))
    alpha = mask.point(lambda value: 110 if value else 0)
    shown = Image.composite(overlay, rgb, alpha)
    contour = np.asarray(mask, dtype=np.uint8) > 0
    if contour.any():
        edge = contour & ~(
            np.roll(contour, 1, 0)
            & np.roll(contour, -1, 0)
            & np.roll(contour, 1, 1)
            & np.roll(contour, -1, 1)
        )
        edge_mask = Image.fromarray((edge * 255).astype(np.uint8), mode="L")
        shown.paste((255, 255, 0), mask=edge_mask)
    return shown.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)


def render_patient_block(
    patient: dict[str, object],
    images: list[dict[str, object]],
    registry_by_image: dict[str, dict[str, str]],
    source_by_image: dict[str, Path],
    mask_by_image: dict[str, tuple[Path, int, int]],
) -> Image.Image:
    columns = max(1, min(7, len(images)))
    rows = math.ceil(len(images) / columns)
    height = 86 + rows * (TILE_SIZE + 28)
    block = Image.new("RGB", (PAGE_WIDTH, height), (20, 20, 20))
    draw = ImageDraw.Draw(block)
    draw.text(
        (10, 7),
        f"rank={patient['review_rank']} {patient['person_key']} fold={patient['outer_fold']} "
        f"ref={patient['reference_class']} L0={patient['l0_top1']} E1={patient['e1_top1']}",
        fill=(255, 255, 255),
        font=get_font(18),
    )
    draw.text(
        (10, 36),
        f"reasons={patient['review_reasons']} mask_group={patient['mask_group']} "
        f"positive_images={patient['positive_mask_image_count']}/{patient['image_count']} "
        f"L1_drift={float(patient['probability_l1_drift']):.4f}",
        fill=(220, 220, 220),
        font=get_font(15),
    )
    for index, image_row in enumerate(images):
        image_key = str(image_row["image_key"])
        mask_path, object_count, positive_pixels = mask_by_image[image_key]
        tile = render_overlay(registry_by_image[image_key], source_by_image[image_key], mask_path)
        x = 10 + (index % columns) * (TILE_SIZE + 28)
        y = 80 + (index // columns) * (TILE_SIZE + 28)
        block.paste(tile, (x, y))
        draw.text(
            (x, y + TILE_SIZE + 3),
            f"{index + 1}: obj={object_count} px={positive_pixels}",
            fill=(230, 230, 230),
            font=get_font(13),
        )
    return block


def main() -> int:
    images = [row for row in read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv") if row["include"] == "1"]
    registry_by_image = {row["image_key"]: row for row in images}
    images_by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in images:
        images_by_person[row["person_key"]].append(row)

    masks = read_csv(MASK_MANIFEST)
    mask_by_image = {
        row["image_key"]: (
            safe_project_path(row["mask_path"]),
            int(row["finding_object_count"]),
            int(row["positive_pixels"]),
        )
        for row in masks
    }
    positive_images_by_person = {
        person_key: sum(mask_by_image[row["image_key"]][2] > 0 for row in rows)
        for person_key, rows in images_by_person.items()
    }
    queue = build_case_review_queue(
        read_csv(OOF_DIR / "L0_oof.csv"),
        read_csv(OOF_DIR / "E1_oof.csv"),
        positive_images_by_person,
    )
    write_csv(PATIENT_QUEUE, queue)

    image_queue: list[dict[str, object]] = []
    for patient in queue:
        for row in sorted(images_by_person[str(patient["person_key"])], key=lambda item: item["image_key"]):
            _, object_count, positive_pixels = mask_by_image[row["image_key"]]
            image_queue.append(
                {
                    "review_rank": patient["review_rank"],
                    "person_key": patient["person_key"],
                    "image_key": row["image_key"],
                    "finding_object_count": object_count,
                    "positive_pixels": positive_pixels,
                }
            )
    write_csv(IMAGE_QUEUE, image_queue)

    sources = read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv")
    source_by_image = {row["image_key"]: safe_project_path(row["raw_image_path"]) for row in sources}
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [
        render_patient_block(
            patient,
            [row for row in image_queue if row["person_key"] == patient["person_key"]],
            registry_by_image,
            source_by_image,
            mask_by_image,
        )
        for patient in queue
    ]
    page_paths: list[Path] = []
    for page_index in range(0, len(blocks), PATIENTS_PER_PAGE):
        page_blocks = blocks[page_index : page_index + PATIENTS_PER_PAGE]
        page = Image.new("RGB", (PAGE_WIDTH, sum(block.height for block in page_blocks)), (12, 12, 12))
        y = 0
        for block in page_blocks:
            page.paste(block, (0, y))
            y += block.height
        path = PRIVATE_DIR / f"page_{page_index // PATIENTS_PER_PAGE + 1:02d}.jpg"
        page.save(path, format="JPEG", quality=88, optimize=True)
        page_paths.append(path)

    reasons = Counter(
        reason
        for patient in queue
        for reason in str(patient["review_reasons"]).split(";")
    )
    summary = {
        "experiment_code": "L0-case-review",
        "status": "review_queue_and_private_visuals_rendered",
        "selection_is_post_model_audit_not_model_selection": True,
        "selection_rule": {
            "per_reference_class_l0_corrected_e1_error": 2,
            "per_reference_class_l0_added_error": 2,
            "per_reference_class_largest_probability_l1_drift": 2,
            "per_reference_class_stable_correct_positive_mask_control": 1,
            "per_reference_class_stable_correct_all_zero_mask_control": 1,
            "tie_break": "probability_l1_drift_desc_then_person_key",
        },
        "patients": len(queue),
        "images": len(image_queue),
        "pages": len(page_paths),
        "reason_counts_nonexclusive": dict(sorted(reasons.items())),
        "mask_groups": dict(sorted(Counter(str(row["mask_group"]) for row in queue).items())),
        "patient_queue_sha256": sha256_file(PATIENT_QUEUE),
        "image_queue_sha256": sha256_file(IMAGE_QUEUE),
        "page_manifest": [
            {"page": path.name, "sha256": sha256_file(path)} for path in page_paths
        ],
        "visuals_are_private": True,
        "image_pixels_may_contain_burned_in_identifiers": True,
        "public_outputs_contain_raw_paths": False,
        "overlay": "complete_reviewed_effective_view_rgb_with_red_mask_fill_and_yellow_boundary",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(SUMMARY_PATH), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
