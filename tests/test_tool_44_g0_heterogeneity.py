import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_tool_44_dry_run_validates_frozen_inputs_without_formal_output():
    report = (
        ROOT
        / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports"
        / "h0_g0_heterogeneity/h0_g0_heterogeneity_audit.json"
    )
    existed_before = report.exists()
    completed = subprocess.run(
        [
            sys.executable,
            "tools/44_audit_g0_heterogeneity.py",
            "--config",
            "configs/research/h0_g0_heterogeneity_audit.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "READY"
    assert payload["patients"] == 967
    assert payload["images"] == 4543
    assert report.exists() is existed_before
    assert "raw_image_path" not in completed.stdout.casefold()
