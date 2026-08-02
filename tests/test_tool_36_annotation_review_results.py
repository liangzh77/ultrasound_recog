import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/36_validate_annotation_review_results.py"
SPEC = importlib.util.spec_from_file_location("tool_36_review_results", TOOL_PATH)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_agreement_report_path_is_versioned_by_input_and_config():
    output = TOOL._output_path("a" * 64, "b" * 64)

    assert output.parent == (
        TOOL.PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
    )
    assert output.name == "annotation_review_agreement_aaaaaaaaaaaa_bbbbbbbbbbbb.json"
