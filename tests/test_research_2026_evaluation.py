from copy import deepcopy
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from openpyxl import Workbook

from src.research_2026_evaluation import (
    PredictionData,
    ReferenceRecord,
    evaluate_reference_predictions,
    load_prediction_csv,
    load_reference_workbook,
    sha256_file,
    validate_reference_rows,
)


ROOT = Path(__file__).resolve().parent.parent
TOOL_SPEC = importlib.util.spec_from_file_location(
    "tool_65_evaluate_2026_reference",
    ROOT / "tools/65_evaluate_2026_reference.py",
)
assert TOOL_SPEC and TOOL_SPEC.loader
TOOL_65 = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL_65)


def small_contract():
    config = yaml.safe_load(
        (ROOT / "configs/research/r3_2026_reference_evaluation.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = deepcopy(config["reference_contract"])
    contract["patients"] = 6
    contract["patient_number_max"] = 6
    return contract, deepcopy(config["evaluation"])


def valid_rows():
    return [
        {
            "患者编号": number,
            "参考状态": "可评价",
            "参考诊断": diagnosis,
            "诊断形成方式": "未见AI结果的独立复核",
            "复核时是否已见AI结果": "否",
            "复核者角色": "超声科医师",
            "复核日期": "2026-08-27",
            "备注（不得填写姓名等身份信息）": "",
        }
        for number, diagnosis in enumerate(
            [
                "正常",
                "类风湿性关节炎",
                "痛风性关节炎",
                "脊柱关节炎",
                "骨性关节炎",
                "损伤",
            ],
            start=1,
        )
    ]


def test_validate_reference_rows_requires_complete_unique_consistent_return():
    contract, _ = small_contract()
    records = validate_reference_rows(valid_rows(), contract)

    assert len(records) == 6
    assert records[0].diagnosis == "正常"
    assert records[-1].patient_number == 6

    duplicate = valid_rows()
    duplicate[-1]["患者编号"] = 1
    with pytest.raises(ValueError, match="重复患者编号"):
        validate_reference_rows(duplicate, contract)

    contaminated = valid_rows()
    contaminated[0]["复核时是否已见AI结果"] = "是"
    with pytest.raises(ValueError, match="矛盾"):
        validate_reference_rows(contaminated, contract)

    incomplete = valid_rows()
    incomplete[0]["参考状态"] = ""
    incomplete[0]["复核日期"] = ""
    with pytest.raises(ValueError, match="参考状态"):
        validate_reference_rows(incomplete, contract)


def test_non_evaluable_rows_must_not_leak_into_scored_reference():
    contract, _ = small_contract()
    rows = valid_rows()
    rows[0]["参考状态"] = "信息不足"
    with pytest.raises(ValueError, match="不得填写参考诊断"):
        validate_reference_rows(rows, contract)
    rows[0]["参考诊断"] = ""
    records = validate_reference_rows(rows, contract)
    assert records[0].diagnosis == ""


def test_reference_workbook_reader_requires_exact_machine_headers(tmp_path):
    contract, _ = small_contract()
    path = tmp_path / "reference.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = contract["sheet_name"]
    sheet.append(list(contract["columns"].values()))
    for row in valid_rows():
        sheet.append([row[column] for column in contract["columns"].values()])
    workbook.save(path)

    loaded = load_reference_workbook(path, contract)
    assert len(loaded) == 6
    assert validate_reference_rows(loaded, contract)[2].diagnosis == "痛风性关节炎"

    bad_path = tmp_path / "bad_headers.xlsx"
    sheet["A1"] = "错误患者列"
    workbook.save(bad_path)
    with pytest.raises(ValueError, match="列名或顺序"):
        load_reference_workbook(bad_path, contract)


def test_prediction_loader_verifies_hash_probabilities_and_rankings(tmp_path):
    contract, evaluation = small_contract()
    path = tmp_path / "predictions.csv"
    classes = evaluation["class_order"]
    slugs = evaluation["probability_slugs"]
    rows = []
    for index in range(6):
        probabilities = np.full(6, 0.02)
        probabilities[index] = 0.90
        ranking = np.argsort(probabilities)[::-1]
        row = {
            "hospital_patient_number": index + 1,
            "image_top1": classes[int(ranking[0])],
            "image_top2": classes[int(ranking[1])],
            "image_low_confidence": 0,
            "hierarchical_top1": classes[int(ranking[0])],
            "hierarchical_top2": classes[int(ranking[1])],
            "hierarchical_low_confidence": 0,
        }
        for slug, value in zip(slugs, probabilities, strict=True):
            row[f"image_prob_{slug}"] = value
            row[f"hierarchical_prob_{slug}"] = value
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data = load_prediction_csv(path, sha256_file(path), evaluation, contract["patients"])
    assert data.patient_numbers == tuple(range(1, 7))
    assert np.allclose(data.image_probabilities.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="哈希"):
        load_prediction_csv(path, "0" * 64, evaluation, contract["patients"])
    rows[0]["image_top1"] = "损伤"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Top1"):
        load_prediction_csv(path, sha256_file(path), evaluation, contract["patients"])


def test_frozen_evaluation_reports_both_modes_and_blinding_subgroups():
    contract, evaluation = small_contract()
    evaluation["bootstrap_samples"] = 100
    records = list(validate_reference_rows(valid_rows(), contract))
    records[5] = ReferenceRecord(
        **{
            **records[5].__dict__,
            "diagnosis_basis": "已见AI结果后的复核",
            "ai_visible": "是",
        }
    )
    image = np.full((6, 6), 0.02)
    hierarchical = np.full((6, 6), 0.02)
    for index in range(6):
        image[index, index] = 0.90
        hierarchical[index, index] = 0.90
    image[5] = np.asarray([0.90, 0.02, 0.02, 0.02, 0.02, 0.02])
    image[5] /= image[5].sum()
    predictions = PredictionData(
        patient_numbers=tuple(range(1, 7)),
        image_probabilities=image / image.sum(axis=1, keepdims=True),
        hierarchical_probabilities=hierarchical / hierarchical.sum(axis=1, keepdims=True),
        image_low_confidence=np.asarray([0, 0, 0, 0, 0, 1]),
        hierarchical_low_confidence=np.zeros(6, dtype=np.int64),
        rows=tuple(
            {"hospital_patient_number": str(index)} for index in range(1, 7)
        ),
    )

    report, joined = evaluate_reference_predictions(records, predictions, evaluation)

    assert report["primary"]["patients"] == 6
    assert report["primary"]["image"]["metrics"]["accuracy"] == pytest.approx(5 / 6)
    assert report["primary"]["hierarchical"]["metrics"]["accuracy"] == 1.0
    assert (
        report["primary"]["paired_hierarchical_minus_image_macro_f1"][
            "observed_delta"
        ]
        > 0
    )
    assert report["independent_not_ai_visible_descriptive"]["patients"] == 5
    assert report["ai_visible_descriptive"]["patients"] == 1
    assert len(joined) == 6

    report.update(
        {
            "git_commit": "a" * 40,
            "config_sha256": "b" * 64,
            "prediction_manifest_sha256": "c" * 64,
            "prediction_delivery_sha256": "d" * 64,
            "reference_workbook_sha256": "e" * 64,
        }
    )
    markdown = TOOL_65.render_markdown(report)
    assert "## 逐类结果" in markdown
    assert "## 混淆矩阵" in markdown
    assert "## 低置信度分层" in markdown
    assert "## AI可见性描述性亚组" in markdown
    assert "## 两个输出的正确性重叠" in markdown
    assert "不设置单一机械性能通过线" in markdown
    assert "医院参考回表SHA-256" in markdown
