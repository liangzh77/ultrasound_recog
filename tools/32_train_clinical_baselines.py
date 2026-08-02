"""Run preregistered C0-C4 abnormal-patient clinical five-fold baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_DERIVED_DIR,
    PATIENT_MULTIMODAL_EXPERIMENT_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/c0_c4_clinical_logreg.yaml",
    )
    parser.add_argument(
        "--clinical-table",
        type=Path,
        default=PATIENT_MULTIMODAL_DERIVED_DIR / "clinical_features.csv",
    )
    parser.add_argument("--experiment-dir", type=Path, default=PATIENT_MULTIMODAL_EXPERIMENT_DIR)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    from src.research_clinical import run_clinical_baselines

    results = run_clinical_baselines(
        args.config.resolve(),
        args.clinical_table.resolve(),
        args.experiment_dir.resolve(),
        ROOT,
    )
    compact = {
        code: {
            "macro_f1": item["summary"]["metrics"]["macro_f1"],
            "oof_sha256": item["summary"]["oof_sha256"],
            "summary": item["summary_path"].as_posix(),
        }
        for code, item in results.items()
        if code != "combined_evaluation"
    }
    compact["combined_evaluation_sha256"] = results["combined_evaluation"]["sha256"]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
