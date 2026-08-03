import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def test_g0_dry_run_validates_frozen_data_without_training():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/42_train_normal_abnormal_gate.py",
            "--config",
            "configs/research/g0_roi_normal_abnormal_gate_b2.yaml",
            "--fold",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    contract = json.loads(completed.stdout)

    assert contract["experiment_code"] == "G0"
    assert contract["task_type"] == "binary_normal_abnormal"
    assert contract["outer_fold"] == 0
    assert contract["gate_counts"] == {
        "patients": 967,
        "images": 4543,
        "normal_patients": 200,
        "abnormal_patients": 767,
    }
    assert contract["outer_test_used_for_training_or_early_stopping"] is False
    assert contract["config_sha256"] == (
        "4a06e647e6b1ba5e4223a4ec752110bf71b9bfc257e8b134271c508c9c53ed72"
    )
