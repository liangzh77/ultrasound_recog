"""Validate the hospital reference return and run the frozen aggregate evaluation once."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_2026_evaluation import (  # noqa: E402
    evaluate_reference_predictions,
    load_prediction_csv,
    load_reference_workbook,
    sha256_file,
    validate_reference_rows,
)


DEFAULT_CONFIG = ROOT / "configs/research/r3_2026_reference_evaluation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-workbook", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def git_revision_clean() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Formal 2026 reference evaluation requires a clean Git worktree")
    return revision


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render_markdown(report: dict) -> str:
    primary = report["primary"]
    image = primary["image"]
    hierarchical = primary["hierarchical"]
    reference = report["reference_summary"]

    def metric_line(name: str, item: dict) -> str:
        metrics = item["metrics"]
        interval = item["macro_f1_95_ci"]
        return (
            f"| {name} | {metrics['macro_f1']:.4f} "
            f"[{interval[0]:.4f}, {interval[2]:.4f}] | {metrics['accuracy']:.4f} | "
            f"{metrics['top2_accuracy']:.4f} | {metrics['ece']:.4f} | "
            f"{metrics['multiclass_brier']:.4f} |"
        )

    rows = [
        "# 2026时间外队列冻结评价结果",
        "",
        "- 状态：医院结构化参考诊断已读取，冻结评价已完成",
        "- 边界：研究用途，不授权临床独立诊断",
        "- 规则：同时报告图像单模态和分层多模态，不依据2026结果择优",
        "",
        "## 参考诊断回收",
        "",
        f"- 返回患者：{report['contract']['patients_returned']}；可评价：{primary['patients']}；",
        f"- 未见AI结果的可评价患者：{reference['ai_not_visible_evaluable']}；已见AI结果：{reference['ai_visible_evaluable']}；",
        f"- 状态分布：{json.dumps(reference['status_counts'], ensure_ascii=False)}；",
        f"- 可评价诊断分布：{json.dumps(reference['diagnosis_counts_evaluable'], ensure_ascii=False)}。",
        "",
        "## 主要评价",
        "",
        "| 输出 | macro-F1（95%CI） | Accuracy | Top-2 | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|",
        metric_line("图像单模态", image),
        metric_line("分层多模态", hierarchical),
        "",
        "分层减图像macro-F1的配对bootstrap结果："
        + json.dumps(
            primary["paired_hierarchical_minus_image_macro_f1"],
            ensure_ascii=False,
        )
        + "。",
        "",
        "## 解释限制",
        "",
        "未见AI结果的独立复核亚组与已见AI结果亚组均只作预先规定的描述性报告；不得按亚组或模型结果择优。逐类指标、混淆矩阵、低/高置信亚组、参考诊断来源及完整冻结合同见同目录JSON。",
        "",
        "本结果是合作研究评价，不构成临床独立诊断性能授权。",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered_waiting_for_hospital_reference":
        raise ValueError("R3 evaluation contract is not preregistered")
    reference_path = args.reference_workbook.resolve()
    if args.validate_only:
        raw_rows = load_reference_workbook(reference_path, config["reference_contract"])
        references = validate_reference_rows(raw_rows, config["reference_contract"])
        validation = {
            "status": "PASS",
            "patients": len(references),
            "evaluable_patients": sum(
                record.status == config["reference_contract"]["evaluable_status"]
                for record in references
            ),
            "reference_workbook_sha256": sha256_file(reference_path),
        }
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    revision = git_revision_clean()
    prediction_contract = config["prediction"]
    manifest_path = ROOT / prediction_contract["manifest_path"]
    if sha256_file(manifest_path) != prediction_contract["manifest_sha256"]:
        raise ValueError("Frozen prediction manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("reference_diagnoses_read") is not False
        or manifest.get("route_selection_affected_by_2026") is not False
        or manifest.get("git_commit") != prediction_contract["frozen_prediction_commit"]
    ):
        raise ValueError("Frozen prediction manifest no longer preserves the blinded route")
    if (
        manifest["artifacts"]["hospital_delivery_sha256"]
        != prediction_contract["hospital_delivery_sha256"]
    ):
        raise ValueError("Frozen hospital delivery hash differs between contracts")

    outputs = config["outputs"]
    output_paths = [ROOT / outputs[key] for key in outputs]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("One-time 2026 external evaluation outputs already exist")
    predictions = load_prediction_csv(
        ROOT / prediction_contract["hospital_delivery_path"],
        prediction_contract["hospital_delivery_sha256"],
        config["evaluation"],
        int(config["reference_contract"]["patients"]),
    )
    raw_rows = load_reference_workbook(reference_path, config["reference_contract"])
    references = validate_reference_rows(raw_rows, config["reference_contract"])
    validation = {
        "status": "PASS",
        "patients": len(references),
        "evaluable_patients": sum(
            record.status == config["reference_contract"]["evaluable_status"]
            for record in references
        ),
        "reference_workbook_sha256": sha256_file(reference_path),
    }
    report, joined = evaluate_reference_predictions(
        references,
        predictions,
        config["evaluation"],
    )
    private_path = ROOT / outputs["private_joined_csv"]
    atomic_write_csv(private_path, joined)
    report.update(
        {
            "experiment_code": config["experiment_code"],
            "status": "formal_external_evaluation_complete",
            "git_commit": revision,
            "config_sha256": sha256_file(config_path),
            "prediction_manifest_sha256": sha256_file(manifest_path),
            "prediction_delivery_sha256": prediction_contract["hospital_delivery_sha256"],
            "reference_workbook_sha256": validation["reference_workbook_sha256"],
            "private_joined_sha256": sha256_file(private_path),
            "reference_diagnoses_read": True,
            "route_selection_affected_by_2026": False,
            "patient_level_results_publicly_reported": False,
        }
    )
    json_path = ROOT / outputs["public_report_json"]
    markdown_path = ROOT / outputs["public_report_markdown"]
    atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                **validation,
                "formal_status": report["status"],
                "public_report_json": str(json_path),
                "public_report_markdown": str(markdown_path),
                "public_report_json_sha256": sha256_file(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
