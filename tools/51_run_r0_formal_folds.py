"""Run resumable R0 preparation, formal training, and OOF ROI export by fold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable).resolve()
REPORT_DIR = (
    ROOT
    / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/auto_roi"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    return parser.parse_args()


def run_tool(*arguments: str) -> None:
    command = [str(PYTHON), *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


def load_completed_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    return payload


def main() -> int:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        print(f"R0 fold {fold}: preparing dataset", flush=True)
        run_tool("tools/48_prepare_auto_roi_dataset.py", "--fold", str(fold))

        training_summary_path = REPORT_DIR / f"R0_fold_{fold}_formal_training_summary.json"
        training_summary = load_completed_summary(training_summary_path)
        if training_summary is None:
            print(f"R0 fold {fold}: starting formal training", flush=True)
            run_tool("tools/49_train_auto_roi_detector.py", "--fold", str(fold))
            training_summary = load_completed_summary(training_summary_path)
            if training_summary is None:
                raise RuntimeError(f"fold {fold} training completed without a durable summary")
        else:
            print(f"R0 fold {fold}: reusing completed formal training", flush=True)

        checkpoint = ROOT / training_summary["best_checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        oof_summary_path = REPORT_DIR / f"R0_fold_{fold}_oof_roi_summary.json"
        if oof_summary_path.is_file():
            oof_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
            if oof_summary.get("checkpoint_sha256") == training_summary["best_checkpoint_sha256"]:
                print(f"R0 fold {fold}: reusing matching OOF ROI export", flush=True)
                continue
        print(f"R0 fold {fold}: exporting outer-test OOF ROI", flush=True)
        run_tool(
            "tools/50_predict_auto_roi_oof.py",
            "--fold",
            str(fold),
            "--checkpoint",
            str(checkpoint),
        )
    print("R0 requested folds completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

