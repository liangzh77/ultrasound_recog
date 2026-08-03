from pathlib import Path

import pytest
import yaml

import numpy as np
from sklearn.metrics import roc_auc_score

from src.research_g0_heterogeneity import (
    benjamini_hochberg,
    binary_metrics,
    binary_oof_probabilities,
    fold_identification,
    fold_feature_shifts,
    load_h0_inputs,
    safe_input_summary,
    spearman_association,
    standardized_mean_difference,
    stratified_bootstrap_ci,
    stratified_permutation,
)
from src.research_proxy_audit import ProxyTable


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


def test_bh_and_smd_are_deterministic():
    assert benjamini_hochberg([0.01, 0.04, 0.03]).tolist() == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    assert standardized_mean_difference(np.array([0, 0]), np.array([0, 0])) == 0
    assert standardized_mean_difference(np.array([0, 1]), np.array([3, 4])) > 3


def test_fold_shift_marks_a_deliberately_shifted_feature():
    folds = np.repeat(np.arange(5), 20)
    values = np.zeros((100, 1))
    values[folds == 2] = 5
    table = ProxyTable(
        person_keys=tuple(f"P{i}" for i in range(100)),
        targets=np.tile([0, 1], 50),
        outer_folds=folds,
        features=values,
        feature_names=("signal",),
    )

    rows, summary = fold_feature_shifts(
        {"g": table}, large_smd=0.5, broad_smd=0.25, broad_fraction=0.2, fdr_alpha=0.05
    )

    fold2 = next(row for row in rows if row["outer_fold"] == 2)
    assert fold2["large_shift"] is True
    assert summary["g"]["broad_shift_passed"] is True


def test_binary_oof_and_single_class_fold_reporting():
    rng = np.random.default_rng(4)
    folds = np.repeat(np.arange(5), 20)
    targets = np.tile([0, 1], 50)
    targets[folds == 3] = 1
    features = rng.normal(size=(100, 2))
    features[:, 0] += targets * 2

    probabilities = binary_oof_probabilities(features, targets, folds, seed=7)
    metrics = binary_metrics(targets, probabilities, folds)

    assert np.isfinite(probabilities).all()
    assert metrics["roc_auc"] > 0.7
    assert metrics["fold_roc_auc"]["3"] is None


def test_stratified_permutation_preserves_each_stratum_and_bootstrap_is_valid():
    values = np.arange(20) % 2
    folds = np.repeat(np.arange(5), 4)
    reference = np.tile([0, 0, 1, 1], 5)
    shuffled = stratified_permutation(values, [folds, reference], seed=9)
    for fold in range(5):
        for label in (0, 1):
            selected = (folds == fold) & (reference == label)
            assert sorted(values[selected]) == sorted(shuffled[selected])
    ci = stratified_bootstrap_ci(
        values,
        values.astype(float),
        [folds, reference],
        roc_auc_score,
        count=20,
        seed=3,
    )
    assert ci == [1.0, 1.0, 1.0]


def test_spearman_association_detects_stable_positive_signal():
    rng = np.random.default_rng(11)
    folds = np.repeat(np.arange(5), 20)
    reference = np.tile([0, 1], 50)
    g0 = np.linspace(0, 1, 100)
    proxy = g0 + rng.normal(0, 0.01, 100)
    result = spearman_association(
        g0,
        proxy,
        folds,
        reference,
        seed=13,
        permutations=19,
        bootstrap_samples=50,
        thresholds={"correlation_abs_rho": 0.3, "significance_alpha": 0.05},
    )
    assert result["spearman_rho"] > 0.9
    assert result["same_direction_folds"] == 5
    assert result["passed"] is True


def test_fold_identification_detects_a_deliberately_encoded_fold():
    folds = np.repeat(np.arange(5), 20)
    table = ProxyTable(
        person_keys=tuple(f"P{i}" for i in range(100)),
        targets=np.tile([0, 1], 50),
        outer_folds=folds,
        features=np.column_stack([folds, folds**2]).astype(float),
        feature_names=("fold_signal", "fold_signal_squared"),
    )

    result = fold_identification(
        table,
        seed=21,
        permutations=19,
        auc_threshold=0.65,
        alpha=0.05,
    )

    assert result["macro_auc"] > 0.95
    assert result["passed"] is True
