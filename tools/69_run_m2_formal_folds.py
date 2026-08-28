"""Run resumable M2 semantic-contour folds and evaluate against M0/E1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable).resolve()
CONFIG_PATH = ROOT / "configs/research/m2_semantic_contour_mean_b2.yaml"
REPORT_DIR = (
    ROOT / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds", type=int, nargs="+", choices=range(5), default=list(range(5))
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def run_tool(*arguments: str) -> None:
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def load_reusable_summary(path: Path, project_root: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"COMPLETED", "EARLY_STOPPED"}:
        return None
    if payload.get("outer_test_iterated") is not True:
        return None
    root = project_root.resolve()
    for path_key, hash_key in (
        ("prediction_path", "prediction_sha256"),
        ("best_checkpoint_path", "best_checkpoint_sha256"),
    ):
        relative = Path(payload.get(path_key, ""))
        if not relative or relative.is_absolute():
            return None
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError:
            return None
        if not artifact.is_file() or sha256_file(artifact) != payload.get(hash_key):
            return None
    return payload


def main() -> int:
    args = parse_args()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M2 formal orchestration requires a clean committed worktree")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_code") != "M2" or config.get("status") != "preregistered":
        raise ValueError("Expected the preregistered M2 configuration")
    seed = int(config["seed"])
    for fold in args.folds:
        summary_path = REPORT_DIR / f"M2-fold{fold}-seed{seed + fold}-formal_summary.json"
        if load_reusable_summary(summary_path, ROOT) is not None:
            print(f"M2 fold {fold}: reusing verified formal result", flush=True)
            continue
        resume_path = (
            ROOT
            / f"workspace/experiments/active/exp_2026-07_patient_multimodal_v1/artifacts/M2/fold_{fold}/resume_formal.pt"
        )
        arguments = [
            "tools/26_train_patient_image_mean.py",
            "--config",
            CONFIG_PATH.relative_to(ROOT).as_posix(),
            "--fold",
            str(fold),
        ]
        if resume_path.is_file():
            arguments.append("--resume")
            print(f"M2 fold {fold}: resuming formal training", flush=True)
        else:
            print(f"M2 fold {fold}: starting formal training", flush=True)
        run_tool(*arguments)
        if load_reusable_summary(summary_path, ROOT) is None:
            raise RuntimeError(f"M2 fold {fold} ended without a reusable result")
    all_fold_paths = [REPORT_DIR / "oof" / f"M2_fold{fold}.csv" for fold in range(5)]
    if args.skip_evaluation or not all(path.is_file() for path in all_fold_paths):
        return 0
    print("M2 fivefold: merging and evaluating against M0/E1", flush=True)
    run_tool(
        "tools/25_evaluate_patient_oof.py",
        "--fold-files",
        *(path.relative_to(ROOT).as_posix() for path in all_fold_paths),
        "--merged-output",
        (REPORT_DIR / "oof/M2_oof.csv").relative_to(ROOT).as_posix(),
        "--baseline",
        (REPORT_DIR / "oof/E1_oof.csv").relative_to(ROOT).as_posix(),
        "--output",
        (REPORT_DIR / "oof/M2_oof_evaluation.json").relative_to(ROOT).as_posix(),
        "--bootstrap",
        "2000",
        "--seed",
        str(seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
