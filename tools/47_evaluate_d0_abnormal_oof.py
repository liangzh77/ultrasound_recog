"""Evaluate the preregistered D0 five-fold abnormal-only OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_EXPERIMENT_DIR  # noqa: E402
from src.research_abnormal import load_d0_config  # noqa: E402
from src.research_abnormal_oof import run_d0_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/d0_roi_abnormal_fiveclass_mil_b2.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_d0_config(config_path)
    output = run_d0_evaluation(
        ROOT, config_path, config, PATIENT_MULTIMODAL_EXPERIMENT_DIR
    )
    result = output["report"]["result"]
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "decision": result["gate"]["decision"],
                "macro_f1": result["models"]["D0"]["metrics"]["macro_f1"],
                "delta_vs_e2_conditional": result["d0_minus_e2_conditional"]["observed_delta"],
                "paired_ci": [
                    result["d0_minus_e2_conditional"]["ci_low"],
                    result["d0_minus_e2_conditional"]["ci_high"],
                ],
                "result_sha256": output["result_sha256"],
                "result_path": output["result_path"].relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
