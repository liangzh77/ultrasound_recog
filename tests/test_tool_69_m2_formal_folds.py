import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "tool_69_m2_formal_folds", ROOT / "tools/69_run_m2_formal_folds.py"
)
m2_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(m2_runner)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reusable_m2_summary_requires_verified_outer_prediction_and_checkpoint(tmp_path):
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("prediction", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = tmp_path / "summary.json"
    payload = {
        "status": "EARLY_STOPPED",
        "outer_test_iterated": True,
        "prediction_path": "prediction.csv",
        "prediction_sha256": _sha256(prediction),
        "best_checkpoint_path": "checkpoint.pt",
        "best_checkpoint_sha256": _sha256(checkpoint),
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert m2_runner.load_reusable_summary(summary, tmp_path) == payload
    checkpoint.write_bytes(b"changed")
    assert m2_runner.load_reusable_summary(summary, tmp_path) is None
