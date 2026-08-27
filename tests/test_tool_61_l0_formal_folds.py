import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "tool_61_l0_formal_folds",
    ROOT / "tools/61_run_l0_formal_folds.py",
)
l0_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(l0_runner)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reusable_l0_summary_requires_hashed_prediction_and_checkpoint(tmp_path):
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("known prediction", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"known checkpoint")
    summary_path = tmp_path / "summary.json"
    payload = {
        "status": "EARLY_STOPPED",
        "outer_test_iterated": True,
        "prediction_path": "prediction.csv",
        "prediction_sha256": _sha256(prediction),
        "best_checkpoint_path": "checkpoint.pt",
        "best_checkpoint_sha256": _sha256(checkpoint),
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    assert l0_runner.load_reusable_summary(summary_path, tmp_path) == payload

    prediction.write_text("changed", encoding="utf-8")
    assert l0_runner.load_reusable_summary(summary_path, tmp_path) is None


def test_reusable_l0_summary_rejects_pilot_or_uniterated_outer_test(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"status": "COMPLETED", "outer_test_iterated": False}),
        encoding="utf-8",
    )

    assert l0_runner.load_reusable_summary(summary_path, tmp_path) is None
