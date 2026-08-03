"""Run the preregistered X0 abnormal-patient OOF complementarity audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_EXPERIMENT_DIR  # noqa: E402
from src.research_fusion import load_x0_inputs, run_x0_formal, sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/x0_abnormal_oof_complementarity.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    set_below_normal_priority()
    config_path = args.config.resolve()
    if args.dry_run:
        config, data = load_x0_inputs(config_path, ROOT)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "study_code": config["study_code"],
                    "patients": len(data.person_keys),
                    "outer_folds": sorted(set(int(value) for value in data.outer_folds)),
                    "config_sha256": sha256_file(config_path),
                    "outputs_written": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = run_x0_formal(config_path, ROOT, PATIENT_MULTIMODAL_EXPERIMENT_DIR)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "decision": result["result"]["d0_feasibility_gate"]["decision"],
                "patient_oof_sha256": result["patient_oof_sha256"],
                "result_sha256": result["result_sha256"],
                "mlflow_parent_run_id": result["mlflow_parent_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
