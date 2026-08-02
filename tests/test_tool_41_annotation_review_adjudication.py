import importlib.util
import json
from pathlib import Path
import time

import pytest

from src.research_annotation_review import PUBLIC_REVIEW_FIELDS
from src.research_ledger import sha256_file


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/41_run_annotation_review_adjudication.py"
SPEC = importlib.util.spec_from_file_location("tool_41_adjudication", TOOL_PATH)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _config():
    return {
        "status": "frozen_preregistered",
        "study": "knee_annotation_review_queue_s1a",
        "dataset_fingerprint": "a" * 64,
        "annotation_version": "annotations_aaaaaaaaaaaa",
        "frozen_provenance": {
            "clinical_confirmation_report_sha256": "b" * 64,
            "queue_sha256": "c" * 64,
            "queue_rows": 1,
            "review_workflow_git_commit": "d" * 40,
        },
        "required_independent_reviews": 2,
        "selection": {
            "diagnosis_visible_to_reviewer": False,
            "legacy_annotation_visible_to_reviewer": False,
        },
        "review_protocol": {
            "stage": "semantic_presence_only",
            "legacy_polygon_visible_to_reviewer": False,
            "reviewer_polygon_capture": False,
            "polygon_action_policy": "fixed_not_applicable",
            "geometry_reliability_stage": "S1b",
        },
        "review_targets": ["积液"],
        "review_fields": {
            "presence_state": ["present", "absent_visible", "not_in_view", "uncertain"],
            "image_mode": ["B", "PD", "CD", "unknown"],
            "annotation_scope": [
                "exhaustive_for_declared_targets",
                "positive_only",
                "uncertain",
            ],
            "polygon_action": ["keep", "adjust", "add", "remove", "not_applicable"],
            "subtype_by_target": {
                "积液": ["joint_effusion", "other_fluid", "uncertain", "not_applicable"]
            },
        },
    }


def _adjudicated_row():
    row = {field: "" for field in PUBLIC_REVIEW_FIELDS}
    row.update(
        {
            "review_case_key": "KNEE_REVIEW_TEST",
            "image_key": "KNEE_IMG_TEST",
            "target_category": "积液",
            "required_independent_reviews": "2",
            "reviewer_1_presence_state": "present",
            "reviewer_2_presence_state": "absent_visible",
            "reviewer_1_image_mode": "B",
            "reviewer_2_image_mode": "B",
            "reviewer_1_annotation_scope": "exhaustive_for_declared_targets",
            "reviewer_2_annotation_scope": "exhaustive_for_declared_targets",
            "reviewer_1_polygon_action": "not_applicable",
            "reviewer_2_polygon_action": "not_applicable",
            "reviewer_1_subtype": "joint_effusion",
            "reviewer_2_subtype": "not_applicable",
            "adjudicated_presence_state": "present",
            "adjudicated_subtype": "joint_effusion",
            "adjudication_notes": "consensus discussion",
        }
    )
    return [row]


def test_merged_manifest_is_hash_and_queue_bound(tmp_path):
    merged = tmp_path / "merged.csv"
    merged.write_text("merged", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_merged_for_adjudication",
        "status": "awaiting_blinded_adjudication",
        "config_sha256": "a" * 64,
        "queue_sha256": "c" * 64,
        "merged_response_sha256": sha256_file(merged),
    }

    TOOL._validate_merged_manifest(manifest, merged, "a" * 64, "c" * 64)
    manifest["queue_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="missing or mismatched"):
        TOOL._validate_merged_manifest(manifest, merged, "a" * 64, "c" * 64)


def test_adjudication_save_writes_complete_privacy_manifest(tmp_path, monkeypatch):
    output = tmp_path / "annotation_review_adjudicated.csv"
    monkeypatch.setattr(
        TOOL,
        "_validate_runtime_git",
        lambda _commit: {"commit": "e" * 40, "dirty": False},
    )

    manifest = TOOL._save_adjudication(
        output,
        _adjudicated_row(),
        _config(),
        "a" * 64,
        "f" * 64,
        time.perf_counter(),
    )

    assert manifest["status"] == "complete"
    assert manifest["adjudicated_response_sha256"] == sha256_file(output)
    assert manifest["privacy"]["adjudicator_name_recorded"] is False
    stored = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert stored["geometry"]["deferred_stage"] == "S1b"


def test_controlled_adjudication_paths_reject_external_file(tmp_path):
    with pytest.raises(ValueError, match="controlled filename"):
        TOOL._controlled_path(
            tmp_path / "adjudicator-name.csv",
            TOOL.REVIEW_DIR,
            "annotation_review_adjudicated.csv",
        )


def test_resume_manifest_rejects_tampered_adjudication(tmp_path):
    output = tmp_path / "annotation_review_adjudicated.csv"
    output.write_text("safe", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "study": "knee_annotation_review_s1a_blinded_adjudication",
        "status": "in_progress",
        "config_sha256": "a" * 64,
        "queue_sha256": "c" * 64,
        "merged_response_sha256": "f" * 64,
        "adjudicated_response_sha256": sha256_file(output),
        "privacy": {"adjudicator_name_recorded": False},
    }
    TOOL._validate_resume_manifest(
        manifest, output, "a" * 64, "c" * 64, "f" * 64
    )
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        TOOL._validate_resume_manifest(
            manifest, output, "a" * 64, "c" * 64, "f" * 64
        )
