"""Prepare one outer fold of the privacy-safe R0 automatic ROI dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import ROOT  # noqa: E402
from src.research_auto_roi import (
    load_r0_config,
    materialize_yolo_dataset,
    read_csv_rows,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/r0_auto_ultrasound_view_roi.yaml"),
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_r0_config((ROOT / args.config).resolve())
    data = config["data"]
    registry_dir = (ROOT / data["registry_dir"]).resolve()
    image_rows = read_csv_rows(registry_dir / data["images_csv"])
    source_rows = read_csv_rows(registry_dir / data["source_map_csv"])
    fold_rows = read_csv_rows(registry_dir / data["inner_folds_csv"])

    source_by_image = {row["image_key"]: row["raw_image_path"] for row in source_rows}
    split_by_person = {
        row["person_key"]: row["split"]
        for row in fold_rows
        if int(row["outer_fold"]) == args.fold
    }
    output_dir = (ROOT / data["output_dir"] / f"fold_{args.fold}").resolve()
    summary = materialize_yolo_dataset(
        root=ROOT,
        image_rows=image_rows,
        source_by_image=source_by_image,
        split_by_person=split_by_person,
        output_dir=output_dir,
    )
    expected = int(data["expected_images"])
    if summary["images"] != expected:
        raise ValueError(f"expected {expected} images, found {summary['images']}")
    print(json.dumps({"fold": args.fold, "output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
