"""Fold-safe, privacy-bounded audit helpers for the fixed G0 OOF result."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.research_proxy_audit import FEATURE_GROUPS, ProxyTable, aggregate_patient_proxy_features


PROBABILITY_COLUMNS = (
    "prob_normal",
    "prob_ra",
    "prob_ga",
    "prob_spa",
    "prob_oa",
    "prob_injury",
)


@dataclass(frozen=True)
class H0Inputs:
    """Validated, pseudonymous patient inputs; no source paths or free text."""

    g0_rows: tuple[dict[str, str], ...]
    proxy_tables: Mapping[str, ProxyTable]
    proxy_oof_rows: Mapping[str, tuple[dict[str, str], ...]]
    input_hashes: Mapping[str, str]
    images: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_input_path(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("H0 input paths must be project-relative")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("H0 input path escapes the project root") from error
    if not path.is_file():
        raise FileNotFoundError("A frozen H0 input is missing")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_file(
    root: Path,
    item: Mapping[str, Any],
    label: str,
) -> tuple[Path, str]:
    path = _safe_input_path(root, str(item["path"]))
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise ValueError(f"Frozen H0 input hash mismatch: {label}")
    return path, actual


def _validate_g0_rows(
    rows: list[dict[str, str]], expected: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    allowed = {
        "person_key",
        "outer_fold",
        "reference_id",
        "predicted_id",
        "prob_abnormal",
        "image_count",
    }
    missing = allowed.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"G0 OOF is missing required columns: {sorted(missing)}")
    if len(rows) != int(expected["patients"]):
        raise ValueError("G0 OOF patient count changed")
    by_person = {row["person_key"]: row for row in rows}
    if len(by_person) != len(rows):
        raise ValueError("G0 OOF patient keys are not unique")
    folds = {int(row["outer_fold"]) for row in rows}
    if folds != set(map(int, expected["folds"])):
        raise ValueError("G0 OOF fold set changed")
    references = np.asarray([int(row["reference_id"]) for row in rows])
    if set(references) != {0, 1}:
        raise ValueError("G0 OOF reference must be binary")
    if int(np.sum(references == 0)) != int(expected["normal_patients"]):
        raise ValueError("G0 OOF normal count changed")
    if int(np.sum(references == 1)) != int(expected["abnormal_patients"]):
        raise ValueError("G0 OOF abnormal count changed")
    probabilities = np.asarray([float(row["prob_abnormal"]) for row in rows])
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("G0 OOF probabilities are invalid")
    return by_person


def load_h0_inputs(root: Path, config: Mapping[str, Any]) -> H0Inputs:
    """Validate frozen hashes and connect only pseudonymous numerical inputs."""
    inputs = config["inputs"]
    hashes: dict[str, str] = {}
    g0_path, hashes["g0_oof"] = _verify_file(root, inputs["g0_oof"], "g0_oof")
    _, hashes["g0_evaluation"] = _verify_file(
        root, inputs["g0_evaluation"], "g0_evaluation"
    )
    feature_path, hashes["image_proxy_features"] = _verify_file(
        root, inputs["image_proxy_features"], "image_proxy_features"
    )
    _, hashes["model_visible_proxy_audit"] = _verify_file(
        root, inputs["model_visible_proxy_audit"], "model_visible_proxy_audit"
    )

    g0_rows = _read_csv(g0_path)
    g0_by_person = _validate_g0_rows(g0_rows, config["expected"])
    image_rows = _read_csv(feature_path)
    if len(image_rows) != int(config["expected"]["images"]):
        raise ValueError("Proxy image count changed")
    if len({row["image_key"] for row in image_rows}) != len(image_rows):
        raise ValueError("Proxy image keys are not unique")

    proxy_tables: dict[str, ProxyTable] = {}
    proxy_oofs: dict[str, tuple[dict[str, str], ...]] = {}
    for group, group_config in config["proxy_groups"].items():
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown frozen proxy group: {group}")
        table = aggregate_patient_proxy_features(image_rows, FEATURE_GROUPS[group])
        if set(table.person_keys) != set(g0_by_person):
            raise ValueError(f"Proxy patient coverage changed: {group}")
        for index, person_key in enumerate(table.person_keys):
            g0_row = g0_by_person[person_key]
            if int(table.outer_folds[index]) != int(g0_row["outer_fold"]):
                raise ValueError(f"Proxy fold mismatch: {group}")
            expected_binary = 0 if int(table.targets[index]) == 0 else 1
            if expected_binary != int(g0_row["reference_id"]):
                raise ValueError(f"Proxy reference mismatch: {group}")
        oof_item = {
            "path": group_config["oof_path"],
            "sha256": group_config["oof_sha256"],
        }
        oof_path, hashes[f"proxy_oof:{group}"] = _verify_file(
            root, oof_item, f"proxy_oof:{group}"
        )
        oof_rows = _read_csv(oof_path)
        if len(oof_rows) != len(g0_rows):
            raise ValueError(f"Proxy OOF patient count changed: {group}")
        oof_by_person = {row["person_key"]: row for row in oof_rows}
        if set(oof_by_person) != set(g0_by_person):
            raise ValueError(f"Proxy OOF coverage changed: {group}")
        for person_key, row in oof_by_person.items():
            g0_row = g0_by_person[person_key]
            probabilities = np.asarray([float(row[name]) for name in PROBABILITY_COLUMNS])
            if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Proxy OOF probabilities invalid: {group}")
            if int(row["outer_fold"]) != int(g0_row["outer_fold"]):
                raise ValueError(f"Proxy OOF fold mismatch: {group}")
            expected_binary = 0 if int(row["reference_id"]) == 0 else 1
            if expected_binary != int(g0_row["reference_id"]):
                raise ValueError(f"Proxy OOF reference mismatch: {group}")
        proxy_tables[group] = table
        proxy_oofs[group] = tuple(oof_by_person[key] for key in sorted(oof_by_person))

    return H0Inputs(
        g0_rows=tuple(g0_by_person[key] for key in sorted(g0_by_person)),
        proxy_tables=proxy_tables,
        proxy_oof_rows=proxy_oofs,
        input_hashes=hashes,
        images=len(image_rows),
    )


def safe_input_summary(inputs: H0Inputs) -> dict[str, Any]:
    return {
        "status": "READY",
        "patients": len(inputs.g0_rows),
        "images": inputs.images,
        "folds": sorted({int(row["outer_fold"]) for row in inputs.g0_rows}),
        "proxy_groups": sorted(inputs.proxy_tables),
        "input_hashes": dict(sorted(inputs.input_hashes.items())),
        "privacy_boundary": "pseudonymous_numerical_inputs_only",
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg q-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("BH p-values must be a finite vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("BH p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute SMD using the pooled population variance."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    pooled = np.sqrt((left.var(ddof=0) + right.var(ddof=0)) / 2.0)
    difference = abs(float(left.mean() - right.mean()))
    if pooled == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / float(pooled)


def fold_feature_shifts(
    proxy_tables: Mapping[str, ProxyTable],
    *,
    large_smd: float,
    broad_smd: float,
    broad_fraction: float,
    fdr_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compare every fold with the other folds and apply one global BH family."""
    from scipy.stats import ks_2samp

    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for group, table in proxy_tables.items():
        for fold in sorted(np.unique(table.outer_folds)):
            selected = table.outer_folds == fold
            for column, feature_name in enumerate(table.feature_names):
                inside = table.features[selected, column]
                outside = table.features[~selected, column]
                ks = ks_2samp(inside, outside, alternative="two-sided", method="auto")
                row = {
                    "proxy_group": group,
                    "outer_fold": int(fold),
                    "feature": feature_name,
                    "fold_mean": float(np.mean(inside)),
                    "fold_median": float(np.median(inside)),
                    "fold_q25": float(np.quantile(inside, 0.25)),
                    "fold_q75": float(np.quantile(inside, 0.75)),
                    "rest_mean": float(np.mean(outside)),
                    "rest_median": float(np.median(outside)),
                    "rest_q25": float(np.quantile(outside, 0.25)),
                    "rest_q75": float(np.quantile(outside, 0.75)),
                    "abs_smd": standardized_mean_difference(inside, outside),
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                }
                rows.append(row)
                p_values.append(float(ks.pvalue))
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["ks_q_value"] = float(q_value)
        row["large_shift"] = bool(
            row["abs_smd"] >= large_smd and q_value <= fdr_alpha
        )
        row["moderate_significant_shift"] = bool(
            row["abs_smd"] >= broad_smd and q_value <= fdr_alpha
        )

    summaries: dict[str, dict[str, Any]] = {}
    for group in proxy_tables:
        group_rows = [row for row in rows if row["proxy_group"] == group]
        shifted = sum(bool(row["moderate_significant_shift"]) for row in group_rows)
        fraction = shifted / len(group_rows)
        summaries[group] = {
            "feature_fold_tests": len(group_rows),
            "large_shift_tests": sum(bool(row["large_shift"]) for row in group_rows),
            "moderate_significant_shift_tests": shifted,
            "moderate_significant_shift_fraction": fraction,
            "broad_shift_passed": bool(fraction >= broad_fraction),
        }
    return rows, summaries


def _pipeline(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def binary_oof_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    outer_folds: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit preprocessing and binary LR inside each fixed G0 outer split."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    probabilities = np.full(len(targets), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(outer_folds)):
        test = outer_folds == fold
        train = ~test
        if len(np.unique(targets[train])) != 2:
            raise ValueError(f"Binary audit training partition is single class: fold {fold}")
        model = _pipeline(seed + int(fold))
        model.fit(features[train], targets[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Binary audit OOF is incomplete")
    return probabilities


def fold_identification(
    table: ProxyTable,
    *,
    seed: int,
    permutations: int,
    auc_threshold: float,
    alpha: float,
) -> dict[str, Any]:
    """Test whether proxy features identify the fixed outer-fold assignment."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    labels = table.outer_folds.astype(np.int64)

    def score(candidate: np.ndarray, run_seed: int) -> float:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
        probabilities = np.full((len(candidate), 5), np.nan, dtype=np.float64)
        for split, (train, test) in enumerate(
            splitter.split(table.features, candidate)
        ):
            model = _pipeline(run_seed + split)
            model.fit(table.features[train], candidate[train])
            classes = model.named_steps["classifier"].classes_.astype(int)
            probabilities[np.ix_(test, classes)] = model.predict_proba(
                table.features[test]
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Fold-identification OOF is incomplete")
        return float(
            roc_auc_score(
                candidate,
                probabilities,
                labels=np.arange(5),
                multi_class="ovr",
                average="macro",
            )
        )

    observed = score(labels, seed)
    null_scores = []
    for index in range(permutations):
        permuted = stratified_permutation(labels, [table.targets], seed=seed + index)
        null_scores.append(score(permuted, seed + index))
    p_value = (sum(value >= observed for value in null_scores) + 1) / (
        permutations + 1
    )
    return {
        "macro_auc": observed,
        "permutation_p_value": float(p_value),
        "permutation_samples": permutations,
        "null_auc_mean": float(np.mean(null_scores)),
        "passed": bool(observed >= auc_threshold and p_value <= alpha),
    }


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    outer_folds: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= 0.5).astype(np.int64)
    fold_auc: dict[str, float | None] = {}
    for fold in sorted(np.unique(outer_folds)):
        selected = np.asarray(outer_folds) == fold
        fold_auc[str(int(fold))] = (
            float(roc_auc_score(targets[selected], probabilities[selected]))
            if len(np.unique(targets[selected])) == 2
            else None
        )
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "fold_roc_auc": fold_auc,
    }


def stratified_permutation(
    values: np.ndarray,
    strata: Sequence[Sequence[int] | np.ndarray],
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values)
    arrays = [np.asarray(item) for item in strata]
    if any(item.shape != values.shape for item in arrays):
        raise ValueError("Permutation strata must match the value vector")
    result = values.copy()
    rng = np.random.default_rng(seed)
    keys = np.column_stack(arrays)
    for key in np.unique(keys, axis=0):
        selected = np.all(keys == key, axis=1)
        result[selected] = rng.permutation(values[selected])
    return result


def stratified_bootstrap_ci(
    targets: np.ndarray,
    values: np.ndarray,
    strata: Sequence[Sequence[int] | np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    count: int,
    seed: int,
) -> list[float]:
    targets = np.asarray(targets)
    values = np.asarray(values)
    arrays = [np.asarray(item) for item in strata]
    keys = np.column_stack(arrays)
    groups = [np.flatnonzero(np.all(keys == key, axis=1)) for key in np.unique(keys, axis=0)]
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(count):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        try:
            score = float(metric(targets[indices], values[indices]))
        except ValueError:
            continue
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        raise ValueError("Bootstrap produced no estimable samples")
    return [float(value) for value in np.quantile(scores, [0.025, 0.5, 0.975])]


def error_predictability(
    table: ProxyTable,
    error_targets: np.ndarray,
    binary_reference: np.ndarray,
    *,
    seed: int,
    permutations: int,
    bootstrap_samples: int,
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one preregistered G0 error target for one proxy group."""
    from sklearn.metrics import roc_auc_score

    probabilities = binary_oof_probabilities(
        table.features, error_targets, table.outer_folds, seed=seed
    )
    metrics = binary_metrics(error_targets, probabilities, table.outer_folds)
    ci = stratified_bootstrap_ci(
        error_targets,
        probabilities,
        [table.outer_folds, binary_reference],
        roc_auc_score,
        count=bootstrap_samples,
        seed=seed,
    )
    null_scores: list[float] = []
    for index in range(permutations):
        permuted = stratified_permutation(
            error_targets,
            [table.outer_folds, binary_reference],
            seed=seed + index,
        )
        null_probability = binary_oof_probabilities(
            table.features,
            permuted,
            table.outer_folds,
            seed=seed + index,
        )
        null_scores.append(float(roc_auc_score(permuted, null_probability)))
    p_value = (sum(score >= metrics["roc_auc"] for score in null_scores) + 1) / (
        permutations + 1
    )
    fold_values = list(metrics["fold_roc_auc"].values())
    all_folds_estimable = all(value is not None for value in fold_values)
    passing_folds = sum(
        value is not None and value >= float(thresholds["error_fold_auc"])
        for value in fold_values
    )
    passed = bool(
        metrics["roc_auc"] >= float(thresholds["error_auc"])
        and ci[0] > float(thresholds["error_ci_lower"])
        and p_value <= float(thresholds["significance_alpha"])
        and all_folds_estimable
        and passing_folds >= int(thresholds["error_min_passing_folds"])
    )
    return {
        **metrics,
        "roc_auc_ci95": ci,
        "permutation_p_value": float(p_value),
        "permutation_samples": permutations,
        "null_auc_mean": float(np.mean(null_scores)),
        "all_folds_estimable": all_folds_estimable,
        "folds_passing_auc_threshold": passing_folds,
        "passed": passed,
    }, probabilities


def spearman_association(
    g0_probability: np.ndarray,
    proxy_probability: np.ndarray,
    outer_folds: np.ndarray,
    binary_reference: np.ndarray,
    *,
    seed: int,
    permutations: int,
    bootstrap_samples: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    def correlation(_unused: np.ndarray, values: np.ndarray) -> float:
        return float(spearmanr(g0_probability[_unused.astype(int)], values).statistic)

    rho = float(spearmanr(g0_probability, proxy_probability).statistic)
    indices = np.arange(len(g0_probability))
    ci = stratified_bootstrap_ci(
        indices,
        proxy_probability,
        [outer_folds, binary_reference],
        correlation,
        count=bootstrap_samples,
        seed=seed,
    )
    fold_rho: dict[str, float] = {}
    for fold in sorted(np.unique(outer_folds)):
        selected = outer_folds == fold
        fold_rho[str(int(fold))] = float(
            spearmanr(g0_probability[selected], proxy_probability[selected]).statistic
        )
    exceedances = 0
    for index in range(permutations):
        permuted = stratified_permutation(
            proxy_probability,
            [outer_folds, binary_reference],
            seed=seed + index,
        )
        null_rho = float(spearmanr(g0_probability, permuted).statistic)
        exceedances += abs(null_rho) >= abs(rho)
    p_value = (exceedances + 1) / (permutations + 1)
    direction = np.sign(rho)
    same_direction_folds = int(
        sum(bool(np.sign(value) == direction) for value in fold_rho.values())
    )
    passed = bool(
        abs(rho) >= float(thresholds["correlation_abs_rho"])
        and (ci[0] > 0 or ci[2] < 0)
        and p_value <= float(thresholds["significance_alpha"])
        and same_direction_folds >= 3
    )
    return {
        "spearman_rho": rho,
        "rho_ci95": ci,
        "fold_spearman_rho": fold_rho,
        "permutation_p_value": float(p_value),
        "permutation_samples": permutations,
        "same_direction_folds": same_direction_folds,
        "passed": passed,
    }
