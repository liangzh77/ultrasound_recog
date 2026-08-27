"""Strict reference-return validation and frozen 2026 aggregate evaluation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from openpyxl import load_workbook

from src.research_metrics import (
    bootstrap_macro_f1_ci,
    compute_patient_metrics,
    paired_bootstrap_macro_f1,
)


@dataclass(frozen=True)
class ReferenceRecord:
    patient_number: int
    status: str
    diagnosis: str
    diagnosis_basis: str
    ai_visible: str
    reviewer_role: str
    review_date: str
    note: str


@dataclass(frozen=True)
class PredictionData:
    patient_numbers: tuple[int, ...]
    image_probabilities: np.ndarray
    hierarchical_probabilities: np.ndarray
    image_low_confidence: np.ndarray
    hierarchical_low_confidence: np.ndarray
    rows: tuple[dict[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _patient_number(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("患者编号不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = _text(value)
    if text.isdigit():
        return int(text)
    raise ValueError(f"非法患者编号：{text or '<空>'}")


def _review_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise ValueError("复核日期必须为YYYY-MM-DD") from error


def load_reference_workbook(path: Path, contract: Mapping[str, Any]) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_name = str(contract["sheet_name"])
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"回表缺少工作表：{sheet_name}")
    values = workbook[sheet_name].iter_rows(values_only=True)
    try:
        header_values = next(values)
    except StopIteration as error:
        raise ValueError("参考诊断回表为空") from error
    headers = tuple(_text(value) for value in header_values)
    expected = tuple(str(value) for value in contract["columns"].values())
    if headers != expected:
        raise ValueError("参考诊断回表列名或顺序与冻结合同不一致")
    rows = []
    for values_row in values:
        if not any(value is not None and _text(value) for value in values_row):
            continue
        rows.append(dict(zip(headers, values_row, strict=True)))
    return rows


def validate_reference_rows(
    rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, Any],
) -> tuple[ReferenceRecord, ...]:
    columns = contract["columns"]
    expected_count = int(contract["patients"])
    minimum = int(contract["patient_number_min"])
    maximum = int(contract["patient_number_max"])
    expected_numbers = set(range(minimum, maximum + 1))
    if len(expected_numbers) != expected_count:
        raise ValueError("冻结患者编号范围与患者数不一致")
    if len(rows) != expected_count:
        raise ValueError(f"回表必须恰好包含{expected_count}行患者记录")

    allowed_statuses = set(contract["allowed_statuses"])
    allowed_diagnoses = set(contract["allowed_diagnoses"])
    allowed_bases = set(contract["allowed_diagnosis_bases"])
    allowed_visibility = set(contract["allowed_ai_visibility"])
    allowed_roles = set(contract["allowed_reviewer_roles"])
    invisible_bases = set(contract["consistency"]["ai_invisible_bases"])
    visible_basis = str(contract["consistency"]["ai_visible_basis"])
    evaluable_status = str(contract["evaluable_status"])
    maximum_note_characters = int(contract["maximum_note_characters"])

    records = []
    for row in rows:
        number = _patient_number(row.get(columns["patient_number"]))
        status = _text(row.get(columns["status"]))
        diagnosis = _text(row.get(columns["diagnosis"]))
        basis = _text(row.get(columns["diagnosis_basis"]))
        ai_visible = _text(row.get(columns["ai_visible"]))
        reviewer_role = _text(row.get(columns["reviewer_role"]))
        review_date = _review_date(row.get(columns["review_date"]))
        note = _text(row.get(columns["note"]))
        prefix = f"患者{number}"
        if status not in allowed_statuses:
            raise ValueError(f"{prefix}参考状态不在冻结值域")
        if basis not in allowed_bases:
            raise ValueError(f"{prefix}诊断形成方式不在冻结值域")
        if ai_visible not in allowed_visibility:
            raise ValueError(f"{prefix}AI结果可见性不在冻结值域")
        if reviewer_role not in allowed_roles:
            raise ValueError(f"{prefix}复核者角色不在冻结值域")
        if status == evaluable_status and diagnosis not in allowed_diagnoses:
            raise ValueError(f"{prefix}可评价但缺少合法六分类参考诊断")
        if status != evaluable_status and diagnosis:
            raise ValueError(f"{prefix}非评价状态不得填写参考诊断")
        if basis in invisible_bases and ai_visible != "否":
            raise ValueError(f"{prefix}诊断形成方式与AI可见性矛盾")
        if basis == visible_basis and ai_visible != "是":
            raise ValueError(f"{prefix}诊断形成方式与AI可见性矛盾")
        if basis == "其他（备注说明）" and not note:
            raise ValueError(f"{prefix}选择其他诊断形成方式时必须备注")
        if len(note) > maximum_note_characters:
            raise ValueError(f"{prefix}备注超过{maximum_note_characters}字")
        records.append(
            ReferenceRecord(
                patient_number=number,
                status=status,
                diagnosis=diagnosis,
                diagnosis_basis=basis,
                ai_visible=ai_visible,
                reviewer_role=reviewer_role,
                review_date=review_date,
                note=note,
            )
        )
    numbers = [record.patient_number for record in records]
    if len(numbers) != len(set(numbers)):
        raise ValueError("回表包含重复患者编号")
    if set(numbers) != expected_numbers:
        raise ValueError("回表患者编号必须完整覆盖冻结范围")
    return tuple(sorted(records, key=lambda record: record.patient_number))


def load_prediction_csv(
    path: Path,
    expected_sha256: str,
    evaluation: Mapping[str, Any],
    expected_patients: int,
) -> PredictionData:
    if sha256_file(path) != expected_sha256:
        raise ValueError("冻结医院预测CSV哈希不一致")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_patients:
        raise ValueError("冻结医院预测CSV患者数不一致")
    slugs = tuple(evaluation["probability_slugs"])
    class_names = tuple(evaluation["class_order"])
    required = {
        "hospital_patient_number",
        "image_top1",
        "image_top2",
        "image_low_confidence",
        "hierarchical_top1",
        "hierarchical_top2",
        "hierarchical_low_confidence",
        *(f"image_prob_{slug}" for slug in slugs),
        *(f"hierarchical_prob_{slug}" for slug in slugs),
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("冻结医院预测CSV缺少必需列")
    numbers = tuple(_patient_number(row["hospital_patient_number"]) for row in rows)
    if len(numbers) != len(set(numbers)):
        raise ValueError("冻结医院预测CSV患者编号重复")
    if set(numbers) != set(range(1, expected_patients + 1)):
        raise ValueError("冻结医院预测CSV患者编号覆盖不完整")
    image = np.asarray(
        [[float(row[f"image_prob_{slug}"]) for slug in slugs] for row in rows],
        dtype=np.float64,
    )
    hierarchical = np.asarray(
        [[float(row[f"hierarchical_prob_{slug}"]) for slug in slugs] for row in rows],
        dtype=np.float64,
    )
    for name, probabilities in (("图像", image), ("分层", hierarchical)):
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{name}冻结概率非法")
    for index, row in enumerate(rows):
        for prefix, probabilities in (
            ("image", image[index]),
            ("hierarchical", hierarchical[index]),
        ):
            ranking = np.argsort(probabilities)[::-1]
            if row[f"{prefix}_top1"] != class_names[int(ranking[0])]:
                raise ValueError(f"{prefix} Top1与冻结概率不一致")
            if row[f"{prefix}_top2"] != class_names[int(ranking[1])]:
                raise ValueError(f"{prefix} Top2与冻结概率不一致")
    image_low = np.asarray([int(row["image_low_confidence"]) for row in rows], dtype=np.int64)
    hierarchical_low = np.asarray(
        [int(row["hierarchical_low_confidence"]) for row in rows], dtype=np.int64
    )
    if not set(image_low).issubset({0, 1}) or not set(hierarchical_low).issubset({0, 1}):
        raise ValueError("冻结低置信标记非法")
    return PredictionData(
        patient_numbers=numbers,
        image_probabilities=image,
        hierarchical_probabilities=hierarchical,
        image_low_confidence=image_low,
        hierarchical_low_confidence=hierarchical_low,
        rows=tuple(rows),
    )


def _descriptive_model_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    calibration_bins: int,
) -> dict[str, Any] | None:
    if len(targets) == 0:
        return None
    return compute_patient_metrics(
        targets,
        probabilities,
        class_names,
        calibration_bins=calibration_bins,
    )


def _model_evaluation(
    targets: np.ndarray,
    probabilities: np.ndarray,
    low_confidence: np.ndarray,
    evaluation: Mapping[str, Any],
    include_interval: bool,
) -> dict[str, Any]:
    class_names = tuple(evaluation["class_order"])
    calibration_bins = int(evaluation["calibration_bins"])
    result: dict[str, Any] = {
        "metrics": compute_patient_metrics(
            targets,
            probabilities,
            class_names,
            calibration_bins=calibration_bins,
        ),
        "low_confidence_patients": int(low_confidence.sum()),
        "high_confidence_patients": int((low_confidence == 0).sum()),
        "low_confidence_metrics": _descriptive_model_metrics(
            targets[low_confidence == 1],
            probabilities[low_confidence == 1],
            class_names,
            calibration_bins,
        ),
        "high_confidence_metrics": _descriptive_model_metrics(
            targets[low_confidence == 0],
            probabilities[low_confidence == 0],
            class_names,
            calibration_bins,
        ),
    }
    if include_interval:
        result["macro_f1_95_ci"] = list(
            bootstrap_macro_f1_ci(
                targets,
                probabilities,
                n_bootstrap=int(evaluation["bootstrap_samples"]),
                seed=int(evaluation["bootstrap_seed"]),
            )
        )
    return result


def evaluate_reference_predictions(
    references: Sequence[ReferenceRecord],
    predictions: PredictionData,
    evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_by_number = {record.patient_number: record for record in references}
    if set(reference_by_number) != set(predictions.patient_numbers):
        raise ValueError("参考诊断与冻结预测患者集合不一致")
    prediction_index = {
        number: index for index, number in enumerate(predictions.patient_numbers)
    }
    ordered_references = [reference_by_number[number] for number in predictions.patient_numbers]
    evaluable_status = "可评价"
    class_names = tuple(evaluation["class_order"])
    class_to_id = {name: index for index, name in enumerate(class_names)}
    primary_indices = np.asarray(
        [
            index
            for index, record in enumerate(ordered_references)
            if record.status == evaluable_status
        ],
        dtype=np.int64,
    )
    if primary_indices.size == 0:
        raise ValueError("没有可评价参考诊断")

    def subgroup(name: str, selected: np.ndarray, include_interval: bool) -> dict[str, Any]:
        selected_targets = np.asarray(
            [class_to_id[ordered_references[index].diagnosis] for index in selected],
            dtype=np.int64,
        )
        if len(selected) == 0:
            return {"name": name, "patients": 0, "status": "not_available"}
        image_probabilities = predictions.image_probabilities[selected]
        hierarchical_probabilities = predictions.hierarchical_probabilities[selected]
        return {
            "name": name,
            "patients": int(len(selected)),
            "class_support": {
                class_name: int(np.sum(selected_targets == class_id))
                for class_id, class_name in enumerate(class_names)
            },
            "image": _model_evaluation(
                selected_targets,
                image_probabilities,
                predictions.image_low_confidence[selected],
                evaluation,
                include_interval,
            ),
            "hierarchical": _model_evaluation(
                selected_targets,
                hierarchical_probabilities,
                predictions.hierarchical_low_confidence[selected],
                evaluation,
                include_interval,
            ),
            "paired_hierarchical_minus_image_macro_f1": (
                paired_bootstrap_macro_f1(
                    selected_targets,
                    image_probabilities,
                    hierarchical_probabilities,
                    n_bootstrap=int(evaluation["bootstrap_samples"]),
                    seed=int(evaluation["bootstrap_seed"]),
                )
                if include_interval
                else None
            ),
        }

    independent_indices = np.asarray(
        [
            index
            for index in primary_indices
            if ordered_references[index].ai_visible == "否"
        ],
        dtype=np.int64,
    )
    ai_visible_indices = np.asarray(
        [
            index
            for index in primary_indices
            if ordered_references[index].ai_visible == "是"
        ],
        dtype=np.int64,
    )
    primary = subgroup("all_evaluable_reference_rows", primary_indices, True)
    independent = subgroup("evaluable_and_ai_not_visible", independent_indices, False)
    visible = subgroup("evaluable_and_ai_visible", ai_visible_indices, False)
    targets = np.asarray(
        [class_to_id[ordered_references[index].diagnosis] for index in primary_indices],
        dtype=np.int64,
    )
    image_correct = predictions.image_probabilities[primary_indices].argmax(axis=1) == targets
    hierarchical_correct = (
        predictions.hierarchical_probabilities[primary_indices].argmax(axis=1) == targets
    )
    primary["correctness_overlap"] = {
        "both_correct": int(np.sum(image_correct & hierarchical_correct)),
        "image_only_correct": int(np.sum(image_correct & ~hierarchical_correct)),
        "hierarchical_only_correct": int(np.sum(~image_correct & hierarchical_correct)),
        "both_wrong": int(np.sum(~image_correct & ~hierarchical_correct)),
    }
    status_counts = Counter(record.status for record in references)
    basis_counts = Counter(record.diagnosis_basis for record in references)
    diagnosis_counts = Counter(
        record.diagnosis for record in references if record.status == evaluable_status
    )
    report = {
        "contract": {
            "patients_returned": len(references),
            "classes": list(class_names),
            "bootstrap_samples": int(evaluation["bootstrap_samples"]),
            "bootstrap_seed": int(evaluation["bootstrap_seed"]),
            "calibration_bins": int(evaluation["calibration_bins"]),
            "primary_population": evaluation["primary_population"],
            "model_selection_from_2026_prohibited": bool(
                evaluation["model_selection_from_2026_prohibited"]
            ),
        },
        "reference_summary": {
            "status_counts": dict(status_counts),
            "diagnosis_counts_evaluable": dict(diagnosis_counts),
            "diagnosis_basis_counts": dict(basis_counts),
            "ai_visible_evaluable": int(len(ai_visible_indices)),
            "ai_not_visible_evaluable": int(len(independent_indices)),
        },
        "primary": primary,
        "independent_not_ai_visible_descriptive": independent,
        "ai_visible_descriptive": visible,
        "clinical_use_authorized": False,
    }
    joined = []
    for number in sorted(reference_by_number):
        record = reference_by_number[number]
        prediction_row = predictions.rows[prediction_index[number]]
        joined.append({**asdict(record), **prediction_row})
    return report, joined
