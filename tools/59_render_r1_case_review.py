"""Render private manual-versus-automatic ROI sheets for the frozen R1 review queue."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_auto_roi import sha256_file  # noqa: E402


QUEUE_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_case_review_queue.csv"
IMAGE_QUEUE_PATH = (
    PATIENT_MULTIMODAL_REPORTS_DIR / "auto_roi" / "R1_case_review_images.csv"
)
SOURCE_PATH = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
PRIVATE_OUTPUT_DIR = (
    PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "r1_case_review_visuals"
)
SUMMARY_PATH = (
    PATIENT_MULTIMODAL_REPORTS_DIR
    / "auto_roi"
    / "R1_case_review_visualization_summary.json"
)
TILE_WIDTH = 384
IMAGE_HEIGHT = 288
TILE_HEADER_HEIGHT = 48
PATIENT_HEADER_HEIGHT = 72
GRID_COLUMNS = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def safe_source_path(relative_path: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("R1 review image source escapes the project root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _box(row: dict[str, str], prefix: str) -> tuple[float, float, float, float]:
    return tuple(float(row[f"{prefix}_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))


def render_image_tile(image_path: Path, row: dict[str, str]) -> Image.Image:
    """Draw manual/automatic view boxes and return one fixed-size audit tile."""

    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    draw = ImageDraw.Draw(source)
    line_width = max(4, min(source.size) // 180)
    draw.rectangle(_box(row, "manual_roi"), outline=(0, 255, 0), width=line_width)
    if row["r0_status"] == "detected":
        draw.rectangle(_box(row, "auto_roi"), outline=(255, 0, 0), width=line_width)
    elif row["r0_status"] == "no_detection":
        draw.rectangle(
            (0, 0, source.width - 1, source.height - 1),
            outline=(255, 0, 0),
            width=line_width,
        )
    else:
        raise ValueError(f"unsupported R0 review status: {row['r0_status']}")

    source.thumbnail((TILE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    tile = Image.new(
        "RGB", (TILE_WIDTH, TILE_HEADER_HEIGHT + IMAGE_HEIGHT), color=(20, 20, 20)
    )
    x = (TILE_WIDTH - source.width) // 2
    y = TILE_HEADER_HEIGHT + (IMAGE_HEIGHT - source.height) // 2
    tile.paste(source, (x, y))
    tile_draw = ImageDraw.Draw(tile)
    font = get_font(14)
    iou = row["reference_iou_descriptive"] or "NA"
    tile_draw.text(
        (6, 4),
        f"{row['image_key']}  {row['r0_status']}",
        fill=(255, 255, 255),
        font=font,
    )
    tile_draw.text(
        (6, 25),
        f"IoU(desc)={iou}  GREEN=manual RED=auto/fallback",
        fill=(220, 220, 220),
        font=font,
    )
    return tile


def render_patient_sheet(
    patient: dict[str, str],
    image_rows: list[dict[str, str]],
    source_by_image: dict[str, Path],
) -> Image.Image:
    if not image_rows:
        raise ValueError("R1 review patient has no images")
    row_count = math.ceil(len(image_rows) / GRID_COLUMNS)
    tile_height = TILE_HEADER_HEIGHT + IMAGE_HEIGHT
    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * TILE_WIDTH, PATIENT_HEADER_HEIGHT + row_count * tile_height),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(sheet)
    font = get_font(16)
    draw.text(
        (8, 7),
        (
            f"rank={patient['review_rank']} person={patient['person_key']} "
            f"fold={patient['outer_fold']} reasons={patient['review_reasons']}"
        ),
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (8, 37),
        (
            f"reference={patient['reference_id']} manual={patient['manual_roi_prediction_id']} "
            f"auto={patient['auto_roi_prediction_id']} "
            f"manual_ok={patient['manual_roi_correct']} auto_ok={patient['auto_roi_correct']}"
        ),
        fill=(230, 230, 230),
        font=font,
    )
    for index, row in enumerate(sorted(image_rows, key=lambda item: item["image_key"])):
        image_key = row["image_key"]
        if image_key not in source_by_image:
            raise ValueError(f"R1 review image lacks a private source: {image_key}")
        tile = render_image_tile(source_by_image[image_key], row)
        column = index % GRID_COLUMNS
        grid_row = index // GRID_COLUMNS
        sheet.paste(
            tile,
            (column * TILE_WIDTH, PATIENT_HEADER_HEIGHT + grid_row * tile_height),
        )
    return sheet


def main() -> int:
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    if dirty:
        raise RuntimeError("Formal R1 review rendering requires a clean Git worktree")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    patients = read_csv(QUEUE_PATH)
    image_rows = read_csv(IMAGE_QUEUE_PATH)
    sources = read_csv(SOURCE_PATH)
    if not patients or not image_rows:
        raise ValueError("R1 case review queues cannot be empty")
    source_by_image = {
        row["image_key"]: safe_source_path(row["raw_image_path"]) for row in sources
    }
    if len(source_by_image) != len(sources):
        raise ValueError("private image source manifest contains duplicate image keys")
    images_by_person: dict[str, list[dict[str, str]]] = {}
    review_image_keys = [row["image_key"] for row in image_rows]
    if len(review_image_keys) != len(set(review_image_keys)):
        raise ValueError("R1 image review queue contains duplicate image keys")
    for row in image_rows:
        images_by_person.setdefault(row["person_key"], []).append(row)
    patient_keys = [row["person_key"] for row in patients]
    if len(patient_keys) != len(set(patient_keys)):
        raise ValueError("R1 case review patient queue contains duplicate patients")
    if set(patient_keys) != set(images_by_person):
        raise ValueError("R1 patient and image review queues cover different patients")
    if not set(review_image_keys) <= set(source_by_image):
        raise ValueError("R1 image review queue is not covered by private image sources")

    PRIVATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    rendered_images = 0
    for patient in sorted(patients, key=lambda row: int(row["review_rank"])):
        person_key = patient["person_key"]
        rows = images_by_person[person_key]
        sheet = render_patient_sheet(patient, rows, source_by_image)
        output_path = PRIVATE_OUTPUT_DIR / f"{person_key}.png"
        sheet.save(output_path, format="PNG", optimize=True)
        rendered_images += len(rows)
        manifest_rows.append(
            {
                "review_rank": patient["review_rank"],
                "person_key": person_key,
                "review_reasons": patient["review_reasons"],
                "image_count": len(rows),
                "sheet_name": output_path.name,
                "sheet_sha256": sha256_file(output_path),
            }
        )

    manifest_path = PRIVATE_OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "experiment_code": "R1-case-review-visualization",
        "status": "private_review_sheets_rendered",
        "patients": len(manifest_rows),
        "images": rendered_images,
        "patient_queue_sha256": sha256_file(QUEUE_PATH),
        "image_queue_sha256": sha256_file(IMAGE_QUEUE_PATH),
        "private_manifest_sha256": sha256_file(manifest_path),
        "git_commit": revision,
        "git_dirty_at_render": False,
        "visuals_are_private": True,
        "image_pixels_may_contain_burned_in_identifiers": True,
        "public_summary_contains_raw_paths": False,
        "geometry_gate": None,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(SUMMARY_PATH), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
