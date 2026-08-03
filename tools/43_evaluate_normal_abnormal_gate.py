"""Merge and evaluate the frozen five-fold G0 normal/abnormal gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_attention_audit import (  # noqa: E402
    audit_attention_rows,
    read_attention_files,
)
from src.research_gate import load_gate_config  # noqa: E402
from src.research_gate_oof import (  # noqa: E402
    evaluate_gate_oof,
    load_and_validate_gate_oof,
    merge_gate_oof_fold_files,
    validate_gate_attention_alignment,
    validate_gate_fold_summaries,
)
from src.research_ledger import sha256_file  # noqa: E402


def _default_fold_paths(directory: Path, stem: str, suffix: str) -> list[Path]:
    return [directory / f"{stem}_fold{fold}{suffix}" for fold in range(5)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-files", nargs=5, type=Path)
    parser.add_argument("--attention-files", nargs=5, type=Path)
    parser.add_argument("--summary-files", nargs=5, type=Path)
    parser.add_argument("--registry-dir", type=Path, default=PATIENT_MULTIMODAL_REGISTRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=PATIENT_MULTIMODAL_REPORTS_DIR / "oof")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_gate_config(config_path)
    fold_files = args.fold_files or _default_fold_paths(
        PATIENT_MULTIMODAL_REPORTS_DIR / "oof", "G0", ".csv"
    )
    attention_files = args.attention_files or _default_fold_paths(
        PATIENT_MULTIMODAL_REPORTS_DIR / "attention", "G0", ".csv"
    )
    summary_files = args.summary_files or [
        PATIENT_MULTIMODAL_REPORTS_DIR
        / f"G0-fold{fold}-seed{config['evaluation']['seeds'][fold]}-formal_summary.json"
        for fold in range(5)
    ]
    for path in [*fold_files, *attention_files, *summary_files]:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    fold_contract = validate_gate_fold_summaries(
        summary_files,
        fold_files,
        attention_files,
        config=config,
        config_path=config_path,
        project_root=ROOT,
    )
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("G0 report output must remain inside the project") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merge_gate_oof_fold_files(fold_files, output_dir / "G0_oof.csv")
    data = load_and_validate_gate_oof(merged_path, args.registry_dir, config)
    attention_rows, attention_inputs = read_attention_files(attention_files)
    validate_gate_attention_alignment(attention_rows, data)
    attention_audit = audit_attention_rows(
        attention_rows,
        collapse_threshold=float(config["model"]["attention_collapse_threshold"]),
        max_collapse_rate=float(config["model"]["max_multi_image_collapse_rate"]),
    )
    evaluation = evaluate_gate_oof(data, config, attention_audit)
    merged_sha = sha256_file(merged_path)
    config_sha = sha256_file(config_path)
    for item, path in zip(attention_inputs, attention_files, strict=True):
        item["path"] = path.resolve().relative_to(ROOT).as_posix()
    report = {
        "experiment_code": "G0",
        "status": "evaluated_passed" if (
            evaluation["performance_attention_gate_passed"]
            and fold_contract["resource_recording_gate_passed"]
        ) else "evaluated_failed",
        "data_fingerprint": config["data_fingerprint"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "git_revision": fold_contract["git_revision"],
        "fold_contract": fold_contract,
        "oof": {
            "path": merged_path.relative_to(ROOT).as_posix(),
            "sha256": merged_sha,
        },
        "attention_inputs": attention_inputs,
        "evaluation": evaluation,
        "gates": {
            "performance_attention": evaluation[
                "performance_attention_gate_passed"
            ],
            "resource_recording": fold_contract["resource_recording_gate_passed"],
        },
    }
    report["all_gates_passed"] = all(report["gates"].values())
    report_path = output_dir / (
        f"G0_oof_evaluation_{merged_sha[:12]}_{config_sha[:12]}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path.resolve()}")
    return 0 if report["all_gates_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
