"""Validate and evaluate one patient-level OOF file, optionally against a baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_oof import (  # noqa: E402
    compare_oof_files,
    evaluate_oof_file,
    merge_oof_fold_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--fold-files", type=Path, nargs="+")
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--registry", type=Path, default=PATIENT_MULTIMODAL_REGISTRY_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    if (args.predictions is None) == (args.fold_files is None):
        raise ValueError("Use exactly one of --predictions or --fold-files")
    prediction_path = args.predictions
    if args.fold_files is not None:
        if args.merged_output is None:
            raise ValueError("--merged-output is required with --fold-files")
        prediction_path = merge_oof_fold_files(args.fold_files, args.merged_output)
    report = evaluate_oof_file(
        prediction_path,
        args.registry,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    if args.baseline is not None:
        report["paired_comparison"] = compare_oof_files(
            args.baseline,
            prediction_path,
            args.registry,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
    output = args.output or (
        PATIENT_MULTIMODAL_REPORTS_DIR
        / "oof"
        / f"{prediction_path.stem}_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["contract"], ensure_ascii=False, indent=2))
    print(f"macro_f1={report['metrics']['macro_f1']:.6f}")
    print(f"report={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
