"""Read only the fixed S1a confirmation cells from an XLSX workbook."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any
from xml.etree import ElementTree
import zipfile

from src.research_annotation_confirmation import (
    EXPECTED_MEDICAL_OPTIONS,
    decisions_equivalent,
)


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MAX_SHARED_STRINGS_XML_BYTES = 5_000_000
MAX_SHEET_XML_BYTES = 5_000_000
MAX_TOTAL_PARSED_XML_BYTES = 30_000_000
REQUIRED_SHEETS = {"填写说明", "医学定义确认", "复核参数确认"}
MEDICAL_OPTION_LETTERS = {
    question_id: {
        chr(ord("A") + index): option
        for index, option in enumerate(options)
    }
    for question_id, (_, options) in EXPECTED_MEDICAL_OPTIONS.items()
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _read_bounded(
    archive: zipfile.ZipFile,
    member: str,
    maximum_bytes: int,
) -> bytes:
    try:
        info = archive.getinfo(member)
    except KeyError as error:
        raise ValueError(f"Confirmation workbook is missing {member}") from error
    if info.file_size > maximum_bytes:
        raise ValueError("Confirmation workbook XML exceeds the size limit")
    return archive.read(info)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = _read_bounded(
            archive,
            "xl/sharedStrings.xml",
            MAX_SHARED_STRINGS_XML_BYTES,
        )
    except ValueError as error:
        if "is missing" in str(error):
            return []
        raise
    root = ElementTree.fromstring(payload)
    return [
        "".join(element.text or "" for element in item.findall(f".//{{{MAIN_NS}}}t"))
        for item in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _worksheet_members(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ElementTree.fromstring(
        _read_bounded(archive, "xl/workbook.xml", MAX_SHEET_XML_BYTES)
    )
    relationships = ElementTree.fromstring(
        _read_bounded(
            archive,
            "xl/_rels/workbook.xml.rels",
            MAX_SHEET_XML_BYTES,
        )
    )
    targets = {
        element.attrib["Id"]: element.attrib["Target"]
        for element in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    result = {}
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        name = sheet.attrib["name"]
        relationship_id = sheet.attrib[f"{{{DOC_REL_NS}}}id"]
        target = targets.get(relationship_id)
        if target is None:
            raise ValueError("Confirmation workbook sheet relationship is missing")
        normalized_target = target.replace("\\", "/")
        if normalized_target.startswith("/"):
            member = normalized_target.lstrip("/")
        else:
            member = posixpath.normpath(posixpath.join("xl", normalized_target))
        if PurePosixPath(member).is_absolute() or member.startswith("../"):
            raise ValueError("Confirmation workbook sheet target is unsafe")
        result[name] = member
    missing = sorted(REQUIRED_SHEETS - set(result))
    if missing:
        raise ValueError(
            "Confirmation workbook is missing required sheets: " + ", ".join(missing)
        )
    return result


def _parse_number(text: str) -> int | float:
    try:
        numeric = float(text)
    except ValueError as error:
        raise ValueError("Confirmation workbook contains an invalid number") from error
    return int(numeric) if numeric.is_integer() else numeric


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(
            element.text or "" for element in cell.findall(f".//{{{MAIN_NS}}}t")
        )
    value_element = cell.find(f"{{{MAIN_NS}}}v")
    if value_element is None or value_element.text is None:
        return None
    text = value_element.text
    if cell_type == "s":
        index = int(text)
        if not 0 <= index < len(shared):
            raise ValueError("Confirmation workbook shared string index is invalid")
        return shared[index]
    if cell_type in {"str", "d"}:
        return text
    if cell_type == "b":
        return text == "1"
    if cell_type == "e":
        raise ValueError("Confirmation workbook contains a cell error")
    return _parse_number(text)


def read_xlsx_cells(path: Path) -> dict[str, dict[str, Any]]:
    """Return cell values for the required sheets without formulas or drawings."""
    if not zipfile.is_zipfile(path):
        raise ValueError("Confirmation workbook is not a valid XLSX container")
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet_members = _worksheet_members(archive)
        total_xml_bytes = 0
        result = {}
        for sheet_name in REQUIRED_SHEETS:
            member = sheet_members[sheet_name]
            info = archive.getinfo(member)
            total_xml_bytes += info.file_size
            if total_xml_bytes > MAX_TOTAL_PARSED_XML_BYTES:
                raise ValueError("Confirmation workbook sheets exceed the size limit")
            root = ElementTree.fromstring(
                _read_bounded(archive, member, MAX_SHEET_XML_BYTES)
            )
            result[sheet_name] = {
                cell.attrib["r"]: _cell_value(cell, shared)
                for cell in root.findall(f".//{{{MAIN_NS}}}c")
            }
    return result


def _option_code(question_id: str, value: Any) -> str:
    text = _clean(value)
    allowed = set(MEDICAL_OPTION_LETTERS[question_id].values())
    if text in allowed:
        return text
    match = re.match(r"^([A-Da-d])(?:\s|[.、:：]|$)", text)
    if not match:
        raise ValueError(f"{question_id} final selection must begin with an option letter")
    letter = match.group(1).upper()
    if letter not in MEDICAL_OPTION_LETTERS[question_id]:
        raise ValueError(f"{question_id} final selection letter is invalid")
    return MEDICAL_OPTION_LETTERS[question_id][letter]


def _iso_date(value: Any, label: str) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = date(1899, 12, 30) + timedelta(days=float(value))
        return converted.isoformat()
    text = _clean(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise ValueError(f"{label} must contain a valid date") from error


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean(value).casefold()
    if text in {"是", "true", "yes", "1"}:
        return True
    if text in {"否", "false", "no", "0"}:
        return False
    raise ValueError(f"{label} must contain 是/否")


def _bootstrap_contract(value: Any) -> dict[str, Any]:
    text = _clean(value)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        return decoded
    if "2000" in text and "95" in text and "点估计" in text:
        return {
            "bootstrap_samples": 2000,
            "ci_level": 0.95,
            "gate_basis": "point_estimate",
        }
    raise ValueError("P8 final value must use the recommended text or JSON")


def extract_confirmation_workbook(path: Path) -> dict[str, Any]:
    """Extract privacy-safe confirmation fields from the fixed five-sheet workbook."""
    sheets = read_xlsx_cells(path)
    medical_cells = sheets["医学定义确认"]
    medical = {}
    for index, question_id in enumerate(EXPECTED_MEDICAL_OPTIONS, start=6):
        if _clean(medical_cells.get(f"A{index}")) != question_id:
            raise ValueError("Medical confirmation row IDs differ from the contract")
        medical[question_id] = {
            "status": _clean(medical_cells.get(f"G{index}")),
            "selected_option": _option_code(
                question_id, medical_cells.get(f"H{index}")
            ),
            "operational_definition": _clean(medical_cells.get(f"I{index}")),
            "decision_reason": _clean(medical_cells.get(f"J{index}")),
            "row_role_present": bool(_clean(medical_cells.get(f"K{index}"))),
            "row_date": _iso_date(
                medical_cells.get(f"L{index}"), f"{question_id} date"
            ),
        }

    parameter_cells = sheets["复核参数确认"]
    parameters = {}
    for index in range(6, 14):
        parameter_id = f"P{index - 5}"
        if _clean(parameter_cells.get(f"A{index}")) != parameter_id:
            raise ValueError("Review parameter row IDs differ from the contract")
        raw_value = parameter_cells.get(f"E{index}")
        if parameter_id in {"P1", "P2"}:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{parameter_id} workbook value must be numeric")
            final_value: Any = int(raw_value)
        elif parameter_id in {"P3", "P5", "P6", "P7"}:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{parameter_id} workbook value must be numeric")
            final_value = float(raw_value)
        elif parameter_id == "P4":
            final_value = _boolean(raw_value, parameter_id)
        else:
            final_value = _bootstrap_contract(raw_value)
        parameters[parameter_id] = {
            "status": _clean(parameter_cells.get(f"F{index}")),
            "final_value": final_value,
            "decision_reason": _clean(parameter_cells.get(f"G{index}")),
        }

    intro_cells = sheets["填写说明"]
    return {
        "medical_decisions": medical,
        "review_parameters": parameters,
        "signoffs": {
            "clinical_signoff_present": bool(_clean(intro_cells.get("B19"))),
            "research_signoff_present": bool(_clean(intro_cells.get("B20"))),
            "confirmation_date": _iso_date(
                intro_cells.get("B21"), "Final confirmation date"
            ),
        },
        "privacy": {
            "signoff_names_retained": False,
            "raw_paths_retained": False,
        },
    }


def reconcile_confirmation_workbook(
    workbook: dict[str, Any],
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    """Require every filled workbook decision to match its machine YAML record."""
    expected_date = _clean(confirmation["signoffs"]["confirmation_date"])
    mismatches = []
    for question_id, workbook_decision in workbook["medical_decisions"].items():
        confirmation_decision = confirmation["medical_decisions"][question_id]
        if workbook_decision["status"] not in {"已确认", "confirmed"}:
            mismatches.append(f"{question_id}.status")
        if workbook_decision["selected_option"] != confirmation_decision[
            "selected_option"
        ]:
            mismatches.append(f"{question_id}.selected_option")
        for field in ("operational_definition", "decision_reason"):
            if _clean(workbook_decision[field]) != _clean(
                confirmation_decision[field]
            ):
                mismatches.append(f"{question_id}.{field}")
        if not workbook_decision["row_role_present"]:
            mismatches.append(f"{question_id}.row_role")
        if workbook_decision["row_date"] != expected_date:
            mismatches.append(f"{question_id}.row_date")

    for parameter_id, workbook_parameter in workbook["review_parameters"].items():
        confirmation_parameter = confirmation["review_parameters"][parameter_id]
        if workbook_parameter["status"] not in {"已确认", "confirmed"}:
            mismatches.append(f"{parameter_id}.status")
        if not decisions_equivalent(
            workbook_parameter["final_value"],
            confirmation_parameter["final_value"],
        ):
            mismatches.append(f"{parameter_id}.final_value")
        if _clean(workbook_parameter["decision_reason"]) != _clean(
            confirmation_parameter["decision_reason"]
        ):
            mismatches.append(f"{parameter_id}.decision_reason")

    workbook_signoffs = workbook["signoffs"]
    if not workbook_signoffs["clinical_signoff_present"]:
        mismatches.append("signoffs.clinical")
    if not workbook_signoffs["research_signoff_present"]:
        mismatches.append("signoffs.research")
    if workbook_signoffs["confirmation_date"] != expected_date:
        mismatches.append("signoffs.confirmation_date")
    if mismatches:
        raise ValueError(
            "Completed workbook and confirmation YAML differ in "
            f"{len(mismatches)} fields: {', '.join(mismatches)}"
        )
    return {
        "status": "completed_workbook_matches_confirmation_yaml",
        "medical_rows_matched": len(workbook["medical_decisions"]),
        "review_parameter_rows_matched": len(workbook["review_parameters"]),
        "per_row_roles_and_dates_present": True,
        "dual_signoff_presence_matched": True,
        "privacy_contract_passed": workbook["privacy"],
    }
