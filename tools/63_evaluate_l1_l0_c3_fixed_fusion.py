"""Run the frozen L1 L0/C3 abnormal-condition OOF sensitivity analysis."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_l0_c3_fusion import (  # noqa: E402
    evaluate,
    load_inputs,
    patient_rows,
    sha256_file,
)


CONFIG_PATH = ROOT / "configs/research/l1_l0_c3_fixed_abnormal_fusion.yaml"


def require_clean_git() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("Formal L1 sensitivity analysis requires a clean Git worktree")
    return revision


def main() -> int:
    git_commit = require_clean_git()
    config, data = load_inputs(CONFIG_PATH, ROOT)
    result = evaluate(config, data)
    output_path = ROOT / config["outputs"]["patient_oof_csv"]
    report_path = ROOT / config["outputs"]["result_json"]
    if output_path.exists() or report_path.exists():
        raise FileExistsError("L1 formal outputs already exist and will not be overwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = patient_rows(data)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "completed_sensitivity_analysis_only",
        "study_code": "L1",
        "git_commit": git_commit,
        "git_dirty_at_start": False,
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_sha256": {
            "L0": config["inputs"]["image_oof"]["sha256"],
            "C3": config["inputs"]["clinical_oof"]["sha256"],
        },
        "patient_oof_path": output_path.relative_to(ROOT).as_posix(),
        "patient_oof_sha256": sha256_file(output_path),
        "result": result,
        "interpretation_contract": config["interpretation_contract"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision_gate"]["decision"],
                "l0_conditional_macro_f1": result["models"]["L0_conditional"]["metrics"]["macro_f1"],
                "c3_macro_f1": result["models"]["C3"]["metrics"]["macro_f1"],
                "fusion_macro_f1": result["models"]["L1_fixed_fusion"]["metrics"]["macro_f1"],
                "delta": result["fixed_fusion_minus_c3"]["observed_delta"],
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
