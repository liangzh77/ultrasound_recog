import importlib.util
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/38_validate_annotation_clinical_confirmation.py"
SPEC = importlib.util.spec_from_file_location("tool_38_confirmation", TOOL_PATH)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _write_workbook_container(path: Path, sheets: set[str]) -> None:
    sheet_xml = "".join(
        f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sorted(sheets), start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships"><sheets>{sheet_xml}</sheets></workbook>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)


def test_workbook_contract_requires_all_five_named_sheets(tmp_path):
    workbook = tmp_path / "complete.xlsx"
    _write_workbook_container(workbook, TOOL.REQUIRED_WORKBOOK_SHEETS)

    result = TOOL._validate_workbook_container(workbook)

    assert result["xlsx_container_valid"] is True
    assert set(result["required_sheets_present"]) == TOOL.REQUIRED_WORKBOOK_SHEETS


def test_workbook_contract_rejects_missing_sheet(tmp_path):
    workbook = tmp_path / "incomplete.xlsx"
    sheets = set(TOOL.REQUIRED_WORKBOOK_SHEETS) - {"医学定义确认"}
    _write_workbook_container(workbook, sheets)

    with pytest.raises(ValueError, match="missing required sheets"):
        TOOL._validate_workbook_container(workbook)


def test_validation_output_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError, match="inside the project"):
        TOOL._project_output(tmp_path / "outside.json")
