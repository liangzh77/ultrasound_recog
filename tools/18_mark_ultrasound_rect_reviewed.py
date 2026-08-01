"""Bulk update ultrasound_rect_reviewed flags in raw annotation JSON files."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import RAW_LABEL_DIR


def main():
    parser = argparse.ArgumentParser(description="批量更新 ultrasound_rect_reviewed 标志位")
    parser.add_argument("--reviewed", choices=["true", "false"], default="true")
    parser.add_argument("--only-with-rect", action="store_true", default=True)
    args = parser.parse_args()

    reviewed = args.reviewed == "true"
    updated = 0
    skipped = 0
    failed = 0

    for json_path in RAW_LABEL_DIR.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "ultrasound_rect" not in data:
                skipped += 1
                continue
            if data.get("ultrasound_rect_reviewed") is reviewed:
                skipped += 1
                continue
            data["ultrasound_rect_reviewed"] = reviewed
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
            updated += 1
        except Exception:
            failed += 1

    print(f"reviewed={reviewed} updated={updated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
