import importlib.util
import json
from pathlib import Path

import pytest

from src.research_ledger import sha256_file


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/40_merge_annotation_review_responses.py"
SPEC = importlib.util.spec_from_file_location("tool_40_merge_review", TOOL_PATH)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_source_manifest_requires_complete_hash_bound_response(tmp_path):
    response = tmp_path / "response.csv"
    response.write_text("safe response", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_independent_response",
        "status": "complete",
        "reviewer_slot": 1,
        "config_sha256": "a" * 64,
        "queue_sha256": "b" * 64,
        "response_sha256": sha256_file(response),
    }

    queue_sha = TOOL._validate_source_manifest(
        manifest, response, 1, "a" * 64
    )

    assert queue_sha == "b" * 64
    manifest["status"] = "in_progress"
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        TOOL._validate_source_manifest(manifest, response, 1, "a" * 64)


def test_manifest_loader_and_controlled_path_fail_closed(tmp_path):
    response = tmp_path / "response.csv"
    response.write_text("safe", encoding="utf-8")
    with pytest.raises(ValueError, match="no provenance manifest"):
        TOOL._manifest(response)

    response.with_suffix(".manifest.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    assert TOOL._manifest(response)["schema_version"] == 1
    with pytest.raises(ValueError, match="controlled filename"):
        TOOL._controlled_path(response, "expected.csv")
