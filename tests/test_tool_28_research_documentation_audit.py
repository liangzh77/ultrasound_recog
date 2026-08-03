import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/28_audit_research_documentation.py"
SPEC = importlib.util.spec_from_file_location("tool_28_docs_audit", TOOL_PATH)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_local_markdown_target_handles_fragments_and_angle_paths():
    assert TOOL._local_markdown_target("guide.md#section") == "guide.md"
    assert TOOL._local_markdown_target("<dir/My%20Guide.md>") == "dir/My Guide.md"
    assert TOOL._local_markdown_target("#local") is None
    assert TOOL._local_markdown_target("https://example.com") is None


def test_validate_markdown_links_rejects_broken_relative_target(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("[missing](missing.md)", encoding="utf-8")

    with pytest.raises(ValueError, match="Broken Markdown link"):
        TOOL._validate_markdown_links(doc)


def test_markdown_file_scope_includes_research_project_decisions_and_root_readme():
    relative_files = {path.relative_to(TOOL.ROOT).as_posix() for path in TOOL._markdown_files()}

    assert "README.md" in relative_files
    assert "docs/research/README.md" in relative_files
    assert "docs/project/S1a盲法分歧裁决工程准备结果_2026-08-03.md" in relative_files
    assert "docs/decisions/ADR-012-拆分S1a语义复核与S1b几何可靠性复核.md" in relative_files
