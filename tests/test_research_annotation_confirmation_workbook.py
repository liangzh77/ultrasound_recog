from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

import pytest

from src.research_annotation_confirmation_workbook import (
    extract_confirmation_workbook,
    read_xlsx_cells,
)


SHEET_NAMES = ["填写说明", "医学定义确认", "复核参数确认", "字段代码本", "文献与版本"]
QUESTION_LETTERS = {
    "Q1": "A",
    "Q2": "B",
    "Q3": "A",
    "Q4": "C",
    "Q5": "A",
    "Q6": "A",
    "Q7": "C",
    "Q8": "A",
}


def _inline_cell(reference: str, value: str) -> str:
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def _number_cell(reference: str, value: int | float) -> str:
    return f'<c r="{reference}"><v>{value}</v></c>'


def _sheet_xml(cells: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData><row>{"".join(cells)}</row></sheetData></worksheet>'
    )


def _write_completed_workbook(
    path: Path,
    *,
    question_selection_override: tuple[str, str] | None = None,
    shared_signoff: bool = False,
    missing_signoff_date: bool = False,
) -> None:
    workbook_sheets = "".join(
        f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(SHEET_NAMES, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    relationships = "".join(
        f'<Relationship Id="rId{index}" Type="worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(SHEET_NAMES) + 1)
    )
    relationships_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )
    intro = [
        (
            '<c r="B19" t="s"><v>0</v></c>'
            if shared_signoff
            else _inline_cell("B19", "controlled clinician name")
        ),
        _inline_cell("B20", "controlled study lead name"),
    ]
    if not missing_signoff_date:
        intro.append(_inline_cell("B21", date(2026, 8, 3).isoformat()))
    medical = []
    for row, question_id in enumerate(QUESTION_LETTERS, start=6):
        letter = QUESTION_LETTERS[question_id]
        if question_selection_override and question_id == question_selection_override[0]:
            letter = question_selection_override[1]
        medical.extend(
            [
                _inline_cell(f"A{row}", question_id),
                _inline_cell(f"G{row}", "已确认"),
                _inline_cell(f"H{row}", f"{letter} selected"),
                _inline_cell(f"I{row}", f"definition for {question_id}"),
                _inline_cell(f"J{row}", "accepted after clinical review"),
                _inline_cell(f"K{row}", "clinical role"),
                _inline_cell(f"L{row}", "2026-08-03"),
            ]
        )
    parameter_values = {
        "P1": 400,
        "P2": 2,
        "P3": 1,
        "P4": "是",
        "P5": 0.8,
        "P6": 0.85,
        "P7": 0.6,
        "P8": "2000次；报告95%CI；硬门槛按点估计",
    }
    parameters = []
    for row, (parameter_id, value) in enumerate(parameter_values.items(), start=6):
        value_cell = (
            _number_cell(f"E{row}", value)
            if isinstance(value, (int, float))
            else _inline_cell(f"E{row}", value)
        )
        parameters.extend(
            [
                _inline_cell(f"A{row}", parameter_id),
                value_cell,
                _inline_cell(f"F{row}", "已确认"),
                _inline_cell(f"G{row}", "accepted before review"),
            ]
        )
    payloads = [intro, medical, parameters, [], []]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        for index, cells in enumerate(payloads, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(cells))
        if shared_signoff:
            archive.writestr(
                "xl/sharedStrings.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                "<si><t>controlled clinician name</t></si></sst>",
            )


def test_completed_workbook_extracts_only_privacy_safe_contract(tmp_path):
    workbook = tmp_path / "completed.xlsx"
    _write_completed_workbook(workbook)

    result = extract_confirmation_workbook(workbook)

    assert result["medical_decisions"]["Q2"]["selected_option"] == (
        "pd_cd_separate_synovial_only"
    )
    assert result["review_parameters"]["P3"]["final_value"] == 1.0
    assert result["review_parameters"]["P8"]["final_value"] == {
        "bootstrap_samples": 2000,
        "ci_level": 0.95,
        "gate_basis": "point_estimate",
    }
    assert result["signoffs"] == {
        "clinical_signoff_present": True,
        "research_signoff_present": True,
        "confirmation_date": "2026-08-03",
    }
    assert "controlled clinician name" not in str(result)
    assert result["privacy"]["signoff_names_retained"] is False


def test_xlsx_reader_supports_shared_strings(tmp_path):
    workbook = tmp_path / "shared.xlsx"
    _write_completed_workbook(workbook, shared_signoff=True)

    result = read_xlsx_cells(workbook)

    assert result["填写说明"]["B19"] == "controlled clinician name"


def test_completed_workbook_maps_human_option_letter(tmp_path):
    workbook = tmp_path / "invalid.xlsx"
    _write_completed_workbook(
        workbook, question_selection_override=("Q1", "B")
    )

    result = extract_confirmation_workbook(workbook)

    assert result["medical_decisions"]["Q1"]["selected_option"] == (
        "anatomy_and_abnormal_mixed"
    )


def test_completed_workbook_rejects_unknown_option_letter(tmp_path):
    workbook = tmp_path / "invalid.xlsx"
    _write_completed_workbook(
        workbook, question_selection_override=("Q1", "E")
    )

    with pytest.raises(ValueError, match="must begin with an option letter"):
        extract_confirmation_workbook(workbook)


def test_completed_workbook_rejects_missing_signoff_date(tmp_path):
    workbook = tmp_path / "completed.xlsx"
    _write_completed_workbook(workbook, missing_signoff_date=True)

    with pytest.raises(ValueError, match="Final confirmation date"):
        extract_confirmation_workbook(workbook)
