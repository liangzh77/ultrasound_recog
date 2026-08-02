import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import time

import pytest
import yaml

from src.research_annotation_review import PUBLIC_REVIEW_FIELDS
from src.research_annotation_review_entry import new_reviewer_response_rows
from src.research_ledger import sha256_file


ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = ROOT / "tools/39_run_annotation_review_entry.py"
SPEC = importlib.util.spec_from_file_location("tool_39_review_entry", TOOL_PATH)
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
            "queue_sha256": "b" * 64,
            "queue_rows": 1,
            "preregistration_git_commit": "c" * 40,
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


def _queue():
    row = {field: "" for field in PUBLIC_REVIEW_FIELDS}
    row.update(
        {
            "review_case_key": "KNEE_REVIEW_TEST",
            "image_key": "KNEE_IMG_TEST",
            "target_category": "积液",
            "required_independent_reviews": "2",
        }
    )
    return [row]


def test_controlled_response_path_rejects_names_and_wrong_slot(tmp_path):
    with pytest.raises(ValueError, match="controlled review filename"):
        TOOL._controlled_response_path(tmp_path / "doctor-name.csv", 1)
    with pytest.raises(ValueError, match="controlled review filename"):
        TOOL._controlled_response_path(
            TOOL.REVIEW_DIR / "annotation_review_reviewer_2_response.csv", 1
        )


def test_session_save_writes_hash_bound_privacy_manifest(tmp_path, monkeypatch):
    config = _config()
    rows = new_reviewer_response_rows(_queue(), config, 1)
    rows[0].update(
        {
            "reviewer_1_presence_state": "present",
            "reviewer_1_image_mode": "B",
            "reviewer_1_annotation_scope": "exhaustive_for_declared_targets",
            "reviewer_1_subtype": "joint_effusion",
        }
    )
    output = tmp_path / "annotation_review_reviewer_1_response.csv"

    monkeypatch.setattr(
        TOOL, "_git_state", lambda: {"commit": "c" * 40, "dirty": False}
    )
    manifest = TOOL._save_session(
        output,
        rows,
        config,
        1,
        "a" * 64,
        "b" * 64,
        time.perf_counter(),
    )

    assert manifest["status"] == "complete"
    assert manifest["response_sha256"] == sha256_file(output)
    assert manifest["privacy"]["reviewer_name_recorded"] is False
    assert manifest["geometry"]["deferred_stage"] == "S1b"
    stored = json.loads(TOOL._manifest_path(output).read_text(encoding="utf-8"))
    TOOL._validate_resume_manifest(stored, output, "a" * 64, "b" * 64, 1)


def test_resume_manifest_fails_after_response_tampering(tmp_path, monkeypatch):
    config = _config()
    rows = new_reviewer_response_rows(_queue(), config, 1)
    output = tmp_path / "annotation_review_reviewer_1_response.csv"
    monkeypatch.setattr(
        TOOL, "_git_state", lambda: {"commit": "c" * 40, "dirty": False}
    )
    TOOL._save_session(
        output, rows, config, 1, "a" * 64, "b" * 64, time.perf_counter()
    )
    manifest = json.loads(TOOL._manifest_path(output).read_text(encoding="utf-8"))
    output.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        TOOL._validate_resume_manifest(manifest, output, "a" * 64, "b" * 64, 1)


def test_session_save_rejects_dirty_or_wrong_git(tmp_path, monkeypatch):
    config = _config()
    rows = new_reviewer_response_rows(_queue(), config, 1)
    output = tmp_path / "annotation_review_reviewer_1_response.csv"
    monkeypatch.setattr(
        TOOL, "_git_state", lambda: {"commit": "c" * 40, "dirty": True}
    )

    with pytest.raises(ValueError, match="runtime Git"):
        TOOL._save_session(
            output, rows, config, 1, "a" * 64, "b" * 64, time.perf_counter()
        )
    assert not output.exists()


def test_main_rejects_queue_hash_before_loading_private_sources(tmp_path, monkeypatch):
    config_path = tmp_path / "frozen.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(), allow_unicode=True), encoding="utf-8"
    )
    queue_path = tmp_path / "queue.csv"
    queue_path.write_text("not the frozen queue", encoding="utf-8")
    monkeypatch.setattr(
        TOOL,
        "parse_args",
        lambda: Namespace(
            reviewer_slot=1,
            config=config_path,
            queue=queue_path,
            response=None,
            smoke_test=True,
        ),
    )
    monkeypatch.setattr(
        TOOL, "_controlled_input_path", lambda path, _parent, _pattern: path.resolve()
    )

    with pytest.raises(ValueError, match="queue hash"):
        TOOL.main()


def test_main_refuses_draft_before_reading_queue_or_creating_output(tmp_path, monkeypatch):
    config_path = tmp_path / "draft.yaml"
    config = _config()
    config["status"] = "draft_not_preregistered"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "should-not-exist.csv"
    monkeypatch.setattr(
        TOOL,
        "parse_args",
        lambda: Namespace(
            reviewer_slot=1,
            config=config_path,
            queue=tmp_path / "missing.csv",
            response=output,
            smoke_test=True,
        ),
    )
    monkeypatch.setattr(
        TOOL, "_controlled_input_path", lambda path, _parent, _pattern: path.resolve()
    )

    with pytest.raises(ValueError, match="frozen_preregistered"):
        TOOL.main()
    assert not output.exists()
