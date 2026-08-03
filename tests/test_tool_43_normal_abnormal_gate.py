import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_tool():
    path = ROOT / "tools/43_evaluate_normal_abnormal_gate.py"
    spec = importlib.util.spec_from_file_location("tool_43", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_g0_evaluator_cli_requires_five_files_per_artifact_type():
    tool = _load_tool()
    args = tool.parse_args(
        ["--config", "configs/research/g0_roi_normal_abnormal_gate_b2.yaml"]
    )
    assert args.fold_files is None
    assert args.attention_files is None
    assert args.summary_files is None
