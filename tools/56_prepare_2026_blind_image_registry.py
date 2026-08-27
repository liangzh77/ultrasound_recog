"""Prepare a deidentified image registry for the label-hidden 2026 cohort."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_REGISTRY_DIR, PATIENT_MULTIMODAL_REPORTS_DIR  # noqa: E402
from src.research_auto_roi import sha256_file  # noqa: E402
from src.research_blind_cohort import build_blind_image_registry  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/r2_2026_frozen_inference.yaml"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty registry: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cohort = config["cohort"]
    secret_path = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "identity_hmac.key"
    if not secret_path.is_file():
        raise FileNotFoundError("existing development identity HMAC key is required")
    public_images, private_sources, private_people, summary = build_blind_image_registry(
        (ROOT / cohort["image_root"]).resolve(),
        secret=secret_path.read_bytes(),
        expected_patients=int(cohort["patients"]),
        expected_images=int(cohort["images"]),
        project_root=ROOT,
    )

    public_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026"
    private_dir = PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "r2_2026"
    public_path = public_dir / "2026_images_public.csv"
    private_source_path = private_dir / "2026_image_sources.csv"
    private_people_path = private_dir / "2026_person_identity_map.csv"
    write_csv(public_path, public_images)
    write_csv(private_source_path, private_sources)
    write_csv(private_people_path, private_people)

    output = {
        "experiment_code": "R2",
        "status": "blind_image_registry_prepared",
        **summary,
        "public_manifest": {
            "path": public_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(public_path),
        },
        "private_source_manifest_sha256": sha256_file(private_source_path),
        "private_person_map_sha256": sha256_file(private_people_path),
        "formal_prediction_generated": False,
    }
    summary_path = public_dir / "2026_image_registry_summary.json"
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(summary_path), **output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

