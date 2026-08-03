from pathlib import Path

import pytest
import yaml

from src.research_g0_heterogeneity import load_h0_inputs, safe_input_summary


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "research" / "h0_g0_heterogeneity_audit.yaml"


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_h0_inputs_join_once_without_private_fields():
    inputs = load_h0_inputs(ROOT, _config())
    summary = safe_input_summary(inputs)

    assert summary["patients"] == 967
    assert summary["images"] == 4543
    assert summary["folds"] == [0, 1, 2, 3, 4]
    assert len(summary["proxy_groups"]) == 6
    serialized = str(summary).casefold()
    assert "raw_image_path" not in serialized
    assert "patient_name" not in serialized
    assert "excel" not in serialized


def test_h0_input_hash_mismatch_fails_closed():
    config = _config()
    config["inputs"]["g0_oof"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        load_h0_inputs(ROOT, config)
