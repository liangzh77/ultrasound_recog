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

    def number(value: object, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    def metric_line(name: str, item: dict) -> str:
        metrics = item["metrics"]
        interval = item["macro_f1_95_ci"]
        return (
            f"| {name} | {metrics['macro_f1']:.4f} "
            f"[{interval[0]:.4f}, {interval[2]:.4f}] | {metrics['accuracy']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['top2_accuracy']:.4f} | "
            f"{number(metrics['macro_auc'])} | {metrics['ece']:.4f} | "
            f"{metrics['multiclass_brier']:.4f} |"
        )

    def per_class_rows() -> list[str]:
        rows = []
        for image_row, hierarchical_row in zip(
            image["metrics"]["per_class"],
            hierarchical["metrics"]["per_class"],
            strict=True,
        ):
            rows.append(
                "| {name} | {support} | {image_sensitivity} | {image_f1} | "
                "{image_auc} | {hierarchical_sensitivity} | {hierarchical_f1} | "
                "{hierarchical_auc} |".format(
                    name=image_row["class_name"],
                    support=image_row["support"],
                    image_sensitivity=number(image_row["sensitivity"]),
                    image_f1=number(image_row["f1"]),
                    image_auc=number(image_row["auc"]),
                    hierarchical_sensitivity=number(hierarchical_row["sensitivity"]),
                    hierarchical_f1=number(hierarchical_row["f1"]),
                    hierarchical_auc=number(hierarchical_row["auc"]),
                )
            )
        return rows

    def confusion_section(name: str, metrics: dict) -> list[str]:
        classes = report["contract"]["classes"]
        header = "| 真实\\预测 | " + " | ".join(classes) + " |"
        separator = "|---|" + "---:|" * len(classes)
        rows = [
            "| " + class_name + " | " + " | ".join(str(value) for value in matrix_row) + " |"
            for class_name, matrix_row in zip(
                classes, metrics["confusion_matrix"], strict=True
            )
        ]
        return [f"### {name}", "", header, separator, *rows, ""]

    def confidence_rows(model_name: str, item: dict) -> list[str]:
        rows = []
        for group_name, key in (("低置信", "low_confidence_metrics"), ("非低置信", "high_confidence_metrics")):
            metrics = item[key]
            if metrics is None:
                rows.append(f"| {model_name} | {group_name} | 0 | NA | NA | NA |")
            else:
                rows.append(
                    f"| {model_name} | {group_name} | {metrics['patients']} | "
                    f"{metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} | "
                    f"{metrics['ece']:.4f} |"
                )
        return rows

    def subgroup_rows() -> list[str]:
        rows = []
        for label, key in (
            ("未见AI结果", "independent_not_ai_visible_descriptive"),
            ("已见AI结果", "ai_visible_descriptive"),
        ):
            subgroup = report[key]
            if subgroup.get("status") == "not_available":
                rows.append(f"| {label} | 0 | NA | NA | NA | NA |")
                continue
            rows.append(
                f"| {label} | {subgroup['patients']} | "
                f"{subgroup['image']['metrics']['macro_f1']:.4f} | "
                f"{subgroup['hierarchical']['metrics']['macro_f1']:.4f} | "
                f"{subgroup['image']['metrics']['accuracy']:.4f} | "
                f"{subgroup['hierarchical']['metrics']['accuracy']:.4f} |"
            )
        return rows

    paired = primary["paired_hierarchical_minus_image_macro_f1"]
    overlap = primary["correctness_overlap"]
    provenance = []
    for label, key in (
        ("正式评价Git", "git_commit"),
        ("评价配置SHA-256", "config_sha256"),
        ("冻结预测manifest SHA-256", "prediction_manifest_sha256"),
        ("冻结预测CSV SHA-256", "prediction_delivery_sha256"),
        ("医院参考回表SHA-256", "reference_workbook_sha256"),
    ):
        if report.get(key):
            provenance.append(f"- {label}：`{report[key]}`；")

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
        "| 输出 | macro-F1（95%CI） | Accuracy | Balanced accuracy | Top-2 | Macro-AUC | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_line("图像单模态", image),
        metric_line("分层多模态", hierarchical),
        "",
        (
            "分层减图像macro-F1为"
            f"{paired['observed_delta']:+.4f}，配对bootstrap 95%CI "
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]，"
            f"差值大于0的bootstrap比例为{paired['probability_delta_gt_zero']:.4f}。"
        ),
        "",
        "这里同时保留两个冻结输出，不依据2026结果选择其中一个作为新的模型。",
        "",
        "## 逐类结果",
        "",
        "| 类别 | 支持数 | 图像灵敏度 | 图像F1 | 图像AUC | 分层灵敏度 | 分层F1 | 分层AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *per_class_rows(),
        "",
        "## 混淆矩阵",
        "",
        *confusion_section("图像单模态", image["metrics"]),
        *confusion_section("分层多模态", hierarchical["metrics"]),
        "## 低置信度分层",
        "",
        "| 输出 | 分组 | 患者数 | Macro-F1 | Accuracy | ECE |",
        "|---|---|---:|---:|---:|---:|",
        *confidence_rows("图像单模态", image),
        *confidence_rows("分层多模态", hierarchical),
        "",
        "低置信标记只表示预登记的重点复核队列，不是安全拒识阈值。",
        "",
        "## AI可见性描述性亚组",
        "",
        "| 亚组 | 患者数 | 图像Macro-F1 | 分层Macro-F1 | 图像Accuracy | 分层Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        *subgroup_rows(),
        "",
        "未见AI结果与已见AI结果亚组只作预先规定的描述性报告，不作模型选择或主要人群替代。",
        "",
        "## 两个输出的正确性重叠",
        "",
        f"- 两者均正确：{overlap['both_correct']}；",
        f"- 仅图像单模态正确：{overlap['image_only_correct']}；",
        f"- 仅分层多模态正确：{overlap['hierarchical_only_correct']}；",
        f"- 两者均错误：{overlap['both_wrong']}。",
        "",
        "## 冻结与复现",
        "",
        *provenance,
        f"- Bootstrap：{report['contract']['bootstrap_samples']}次，种子`{report['contract']['bootstrap_seed']}`；",
        f"- 校准分箱：{report['contract']['calibration_bins']}；",
        "- 患者级连接结果保存在受保护目录，公开报告只含聚合统计。",
        "",
        "## 结论边界与医院验收",
        "",
        "本评价不设置单一机械性能通过线，也不自动给出医院项目通过或不通过结论。应结合macro-F1区间、逐类灵敏度/F1、混淆、校准、低置信亚组、参考诊断形成方式和医生实际复核意见，判断全自动研究流程是否达到项目可接受水平。",
        "",
        "若结果不足，不得在同一2026队列上调参后重新称为独立时间外验证；任何半自动补救或新模型都必须另立版本。本结果是合作研究评价，不构成临床独立诊断性能授权。",
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
