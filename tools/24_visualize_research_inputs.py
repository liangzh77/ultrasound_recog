"""Create de-identified full-vs-ROI audit sheets for every fold and class."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_dataset import load_fold_records  # noqa: E402
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402
from src.research_transforms import extract_region, letterbox_rgb  # noqa: E402


OUTPUT_SIZE = 256
HEADER_HEIGHT = 24


def main() -> int:
    output_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "input_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []

    for fold in range(5):
        records = load_fold_records(
            PATIENT_MULTIMODAL_REGISTRY_DIR,
            ROOT,
            outer_fold=fold,
            split="test",
        )
        by_class = defaultdict(list)
        for record in records:
            by_class[record.diagnosis_id].append(record)
        canvas = Image.new(
            "RGB",
            (OUTPUT_SIZE * 2, (OUTPUT_SIZE + HEADER_HEIGHT) * len(DIAGNOSIS_CLASSES)),
            color=(24, 24, 24),
        )
        draw = ImageDraw.Draw(canvas)
        for class_id, diagnosis in enumerate(DIAGNOSIS_CLASSES):
            if not by_class[class_id]:
                raise ValueError(f"Fold {fold} has no image for class {class_id}")
            record = sorted(by_class[class_id], key=lambda item: item.image_key)[0]
            with Image.open(record.image_path) as source:
                full = extract_region(source, record.roi, input_mode="full")
                roi = extract_region(source, record.roi, input_mode="roi")
                full_size = full.size
                roi_size = roi.size
                full_view = letterbox_rgb(full, OUTPUT_SIZE)
                roi_view = letterbox_rgb(roi, OUTPUT_SIZE)
            top = class_id * (OUTPUT_SIZE + HEADER_HEIGHT)
            draw.text((4, top + 5), f"F{fold} C{class_id} FULL", fill=(255, 255, 255))
            draw.text(
                (OUTPUT_SIZE + 4, top + 5),
                f"F{fold} C{class_id} ROI",
                fill=(255, 255, 255),
            )
            canvas.paste(full_view, (0, top + HEADER_HEIGHT))
            canvas.paste(roi_view, (OUTPUT_SIZE, top + HEADER_HEIGHT))
            audit_rows.append(
                {
                    "outer_fold": fold,
                    "diagnosis_id": class_id,
                    "diagnosis": diagnosis,
                    "person_key": record.person_key,
                    "image_key": record.image_key,
                    "full_width": full_size[0],
                    "full_height": full_size[1],
                    "roi_width": roi_size[0],
                    "roi_height": roi_size[1],
                    "roi_area_fraction": round(
                        (roi_size[0] * roi_size[1]) / (full_size[0] * full_size[1]),
                        6,
                    ),
                }
            )
        canvas.save(output_dir / f"fold_{fold}_full_vs_roi.jpg", quality=92)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "sample_count": len(audit_rows),
        "coverage": "one test image per outer fold and diagnosis",
        "metadata_contains_raw_paths_or_names": False,
        "image_pixels_may_contain_burned_in_identifiers": True,
        "rows": audit_rows,
    }
    destination = output_dir / "input_audit.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(destination)
    print(json.dumps({"sample_count": len(audit_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
