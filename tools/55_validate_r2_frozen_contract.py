"""Validate frozen R2 inputs without reading 2026 reference diagnoses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_auto_roi import require_sha256  # noqa: E402
from src.research_frozen_inference import audit_numbered_image_cohort  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research/r2_2026_frozen_inference.yaml"
REPORT_DIR = (
    ROOT
    / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/r2_2026"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_code") != "R2":
        raise ValueError("R2 config must declare experiment_code: R2")

    cohort = config["cohort"]
    image_root = (ROOT / cohort["image_root"]).resolve()
    cohort_audit = audit_numbered_image_cohort(
        image_root,
        expected_patients=int(cohort["patients"]),
        expected_images=int(cohort["images"]),
    )
    workbook = (ROOT / cohort["clinical_workbook"]).resolve()
    workbook_hash = require_sha256(
        workbook, cohort["clinical_workbook_sha256"], "2026 clinical workbook"
    )

    artifact_hashes = []
    for group_name, member_key in (
        ("E1", "image_classifier_ensemble"),
        ("C3", "clinical_ensemble"),
    ):
        members = config[member_key][
            "checkpoints" if group_name == "E1" else "models"
        ]
        for fold, member in sorted(members.items(), key=lambda item: int(item[0])):
            path = (ROOT / member["path"]).resolve()
            actual_hash = require_sha256(path, member["sha256"], f"{group_name} fold {fold}")
            artifact_hashes.append(
                {"group": group_name, "fold": int(fold), "sha256": actual_hash}
            )

    selected_route = config["route"]["selected"]
    if selected_route not in {"pending_R1_development_only", "full_auto", "semi_auto"}:
        raise ValueError(f"unsupported frozen route: {selected_route}")
    if selected_route == "full_auto" and not config["auto_roi_ensemble"].get("checkpoints"):
        raise ValueError("full-auto route requires five frozen R0 checkpoints")

    status = "waiting_for_R1_route" if selected_route.startswith("pending") else "ready"
    summary = {
        "experiment_code": "R2",
        "status": status,
        "selected_route": selected_route,
        "cohort": cohort_audit,
        "clinical_workbook_sha256": workbook_hash,
        "verified_frozen_artifacts": artifact_hashes,
        "verified_frozen_artifact_count": len(artifact_hashes),
        "reference_diagnoses_read": False,
        "formal_prediction_generated": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "R2_frozen_input_preflight.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

