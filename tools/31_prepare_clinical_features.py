"""Create the deidentified, leakage-audited abnormal-patient clinical table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_DERIVED_DIR,
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
    RAW_LABEL_DIR,
)
from src.research_clinical import prepare_clinical_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/c0_c4_clinical_logreg.yaml",
    )
    parser.add_argument("--raw-label-dir", type=Path, default=RAW_LABEL_DIR)
    parser.add_argument("--registry", type=Path, default=PATIENT_MULTIMODAL_REGISTRY_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=PATIENT_MULTIMODAL_DERIVED_DIR / "clinical_features.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PATIENT_MULTIMODAL_REPORTS_DIR / "clinical/clinical_input_audit.json",
    )
    args = parser.parse_args()
    report = prepare_clinical_features(
        args.config.resolve(),
        args.raw_label_dir.resolve(),
        args.registry.resolve(),
        args.output.resolve(),
        args.audit.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
