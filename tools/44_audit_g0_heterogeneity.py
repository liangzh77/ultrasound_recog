"""Run the preregistered CPU-only audit of G0 fold heterogeneity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Set limits before importing NumPy/scikit-learn through project modules.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import PATIENT_MULTIMODAL_EXPERIMENT_DIR  # noqa: E402
from src.research_g0_heterogeneity import (  # noqa: E402
    H0Inputs,
    error_predictability,
    fold_feature_shifts,
    fold_identification,
    load_h0_inputs,
    safe_input_summary,
    sha256_file,
    spearman_association,
)
from src.research_proxy_audit import ProxyTable  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402
from src.research_tracking import LocalResearchTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/h0_g0_heterogeneity_audit.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_output_directory(relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("H0 output directory must be project-relative")
    path = (ROOT / relative_path).resolve()
    reports = (
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "reports"
    ).resolve()
    try:
        path.relative_to(reports)
    except ValueError as error:
        raise ValueError("H0 output directory must stay inside derived reports") from error
    return path


def _subset_table(table: ProxyTable, selected: np.ndarray) -> ProxyTable:
    indices = np.flatnonzero(selected)
    return ProxyTable(
        person_keys=tuple(table.person_keys[index] for index in indices),
        targets=table.targets[selected],
        outer_folds=table.outer_folds[selected],
        features=table.features[selected],
        feature_names=table.feature_names,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("H0 aggregate CSV cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _deadline_guard(started: float, hard_limit_minutes: float) -> None:
    if (time.perf_counter() - started) / 60 >= hard_limit_minutes:
        raise TimeoutError("H0 hard runtime limit reached")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _git_is_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def run_analysis(
    inputs: H0Inputs,
    config: dict[str, object],
    *,
    started: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    settings = config["statistics"]
    assert isinstance(settings, dict)
    seed = int(config["seed"])
    permutations = int(settings["permutations"])
    bootstrap_samples = int(settings["bootstrap_samples"])
    hard_limit = float(config["resources"]["hard_limit_minutes"])

    shift_rows, shift_summary = fold_feature_shifts(
        inputs.proxy_tables,
        large_smd=float(settings["large_shift_smd"]),
        broad_smd=float(settings["broad_shift_smd"]),
        broad_fraction=float(settings["broad_shift_fraction"]),
        fdr_alpha=float(settings["fdr_alpha"]),
    )
    fold_models = {}
    for group_index, (group, table) in enumerate(inputs.proxy_tables.items()):
        _deadline_guard(started, hard_limit)
        print(f"H0-A fold identification: {group}", flush=True)
        fold_models[group] = fold_identification(
            table,
            seed=seed + group_index * 10_000,
            permutations=permutations,
            auc_threshold=float(settings["fold_identification_auc"]),
            alpha=float(settings["significance_alpha"]),
        )
        shift_summary[group]["fold_identification"] = fold_models[group]
        shift_summary[group]["passed"] = bool(
            shift_summary[group]["broad_shift_passed"]
            or fold_models[group]["passed"]
        )

    g0_reference = np.asarray(
        [int(row["reference_id"]) for row in inputs.g0_rows], dtype=np.int64
    )
    g0_predicted = np.asarray(
        [int(row["predicted_id"]) for row in inputs.g0_rows], dtype=np.int64
    )
    g0_probability = np.asarray(
        [float(row["prob_abnormal"]) for row in inputs.g0_rows], dtype=np.float64
    )
    target_specs = {
        "total_error": np.ones(len(g0_reference), dtype=bool),
        "normal_false_positive": g0_reference == 0,
        "abnormal_false_negative": g0_reference == 1,
    }
    error_results: dict[str, dict[str, object]] = {}
    error_oof_rows: list[dict[str, object]] = []
    for group_index, (group, full_table) in enumerate(inputs.proxy_tables.items()):
        error_results[group] = {}
        for target_index, (target_name, selected) in enumerate(target_specs.items()):
            _deadline_guard(started, hard_limit)
            print(f"H0-B error predictability: {group}/{target_name}", flush=True)
            table = _subset_table(full_table, selected)
            errors = (g0_predicted[selected] != g0_reference[selected]).astype(np.int64)
            references = g0_reference[selected]
            result, probabilities = error_predictability(
                table,
                errors,
                references,
                seed=seed + group_index * 100_000 + target_index * 10_000,
                permutations=permutations,
                bootstrap_samples=bootstrap_samples,
                thresholds=settings,
            )
            result["patients"] = len(table.person_keys)
            result["errors"] = int(errors.sum())
            error_results[group][target_name] = result
            for index, person_key in enumerate(table.person_keys):
                error_oof_rows.append(
                    {
                        "person_key": person_key,
                        "outer_fold": int(table.outer_folds[index]),
                        "proxy_group": group,
                        "error_target": target_name,
                        "reference_error": int(errors[index]),
                        "probability_error": float(probabilities[index]),
                    }
                )

    correlation_results: dict[str, dict[str, object]] = {}
    correlation_rows: list[dict[str, object]] = []
    for group_index, group in enumerate(inputs.proxy_tables):
        _deadline_guard(started, hard_limit)
        print(f"H0-C proxy OOF association: {group}", flush=True)
        rows = inputs.proxy_oof_rows[group]
        proxy_probability = 1.0 - np.asarray(
            [float(row["prob_normal"]) for row in rows], dtype=np.float64
        )
        outer_folds = inputs.proxy_tables[group].outer_folds
        result = spearman_association(
            g0_probability,
            proxy_probability,
            outer_folds,
            g0_reference,
            seed=seed + group_index * 100_000,
            permutations=bootstrap_samples,
            bootstrap_samples=bootstrap_samples,
            thresholds=settings,
        )
        correlation_results[group] = result
        correlation_rows.append(
            {
                "proxy_group": group,
                "spearman_rho": result["spearman_rho"],
                "rho_ci95_low": result["rho_ci95"][0],
                "rho_ci95_median": result["rho_ci95"][1],
                "rho_ci95_high": result["rho_ci95"][2],
                "permutation_p_value": result["permutation_p_value"],
                "same_direction_folds": result["same_direction_folds"],
                "passed": result["passed"],
            }
        )

    group_signal = {}
    for group in inputs.proxy_tables:
        error_passed = any(
            bool(result["passed"]) for result in error_results[group].values()
        )
        group_signal[group] = {
            "fold_shift_or_identification_passed": bool(shift_summary[group]["passed"]),
            "any_error_target_passed": error_passed,
            "correlation_passed": bool(correlation_results[group]["passed"]),
            "combined_association_passed": bool(
                error_passed
                and (
                    shift_summary[group]["passed"]
                    or correlation_results[group]["passed"]
                )
            ),
        }
    interpretations = []
    if any(group_signal[group]["combined_association_passed"] for group in ("image_count_only", "roi_aspect_visible")):
        interpretations.append("model_visible_proxy_association")
    if group_signal["roi_resolution_upper_bound"]["combined_association_passed"]:
        interpretations.append("inferable_geometry_association")
    if any(group_signal[group]["combined_association_passed"] for group in ("dimensions_export", "outer_nonmedical")):
        interpretations.append("external_acquisition_export_heterogeneity")
    if group_signal["roi_edge_visible_control"]["combined_association_passed"]:
        interpretations.append("medical_edge_control_association")
    if not interpretations:
        interpretations.append("measured_proxies_did_not_explain")

    report = {
        "audit": "h0_g0_fold_heterogeneity",
        "status": "completed",
        "dataset_fingerprint": config["dataset_fingerprint"],
        "patients": len(inputs.g0_rows),
        "images": inputs.images,
        "folds": sorted({int(row["outer_fold"]) for row in inputs.g0_rows}),
        "seed": seed,
        "input_hashes": dict(inputs.input_hashes),
        "statistics": {
            "permutations": permutations,
            "bootstrap_samples": bootstrap_samples,
            "thresholds": settings,
        },
        "h0_a_fold_shift": shift_summary,
        "h0_b_error_predictability": error_results,
        "h0_c_proxy_probability_association": correlation_results,
        "group_signal": group_signal,
        "interpretations": interpretations,
        "causal_claim": False,
        "new_diagnostic_model_trained": False,
        "persisted_weights_sha256": "not_applicable_no_persisted_weights",
    }
    return report, shift_rows, error_oof_rows, correlation_rows


def main() -> int:
    started = time.perf_counter()
    below_normal = set_below_normal_priority()
    args = parse_args()
    config_path = args.config.resolve()
    try:
        config_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("H0 config must stay inside the project") from error
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = load_h0_inputs(ROOT, config)
    ready = safe_input_summary(inputs)
    ready["config_sha256"] = sha256_file(config_path)
    if args.dry_run:
        print(json.dumps(ready, ensure_ascii=False, indent=2))
        return 0
    if not _git_is_clean():
        raise ValueError("Formal H0 requires a clean Git worktree")

    report, shift_rows, error_rows, correlation_rows = run_analysis(
        inputs, config, started=started
    )
    report["provenance"] = {
        "git_commit": _git_commit(),
        "config_sha256": sha256_file(config_path),
    }
    report["runtime"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numeric_thread_limit": 2,
        "below_normal_priority": below_normal,
        "gpu_used": False,
    }

    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking",
        "patient-normal-abnormal-gate",
    )
    with tracker.parent_run(
        "H0-g0-heterogeneity-formal",
        {
            "study": config["study"],
            "dataset_fingerprint": config["dataset_fingerprint"],
            "config_sha256": report["provenance"]["config_sha256"],
            "git_commit": report["provenance"]["git_commit"],
            "status": "completed",
        },
    ) as run:
        report["tracking"] = {
            "mlflow_experiment": "patient-normal-abnormal-gate",
            "mlflow_parent_run_id": run.info.run_id,
        }
        tracker.log_metrics(
            {
                f"fold_id_auc_{group}": values["fold_identification"]["macro_auc"]
                for group, values in report["h0_a_fold_shift"].items()
            }
        )
        tracker.log_metrics(
            {
                f"error_auc_{group}_total": values["total_error"]["roc_auc"]
                for group, values in report["h0_b_error_predictability"].items()
            }
        )
        tracker.log_metrics(
            {
                f"rho_{group}": values["spearman_rho"]
                for group, values in report["h0_c_proxy_probability_association"].items()
            }
        )

    output_config = config["outputs"]
    output_dir = _safe_output_directory(str(output_config["directory"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    shift_path = output_dir / str(output_config["fold_shift_csv"])
    error_path = output_dir / str(output_config["error_oof_csv"])
    correlation_path = output_dir / str(output_config["correlation_csv"])
    _write_csv(shift_path, shift_rows)
    _write_csv(error_path, error_rows)
    _write_csv(correlation_path, correlation_rows)
    report["artifacts"] = {
        "fold_shift_csv": {
            "path": shift_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(shift_path),
        },
        "error_oof_csv": {
            "path": error_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(error_path),
        },
        "correlation_csv": {
            "path": correlation_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(correlation_path),
        },
    }
    report_path = output_dir / str(output_config["report"])
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    forbidden = tuple(str(token).casefold() for token in config["privacy"]["forbidden_output_tokens"])
    if any(token in serialized.casefold() for token in forbidden):
        raise ValueError("H0 report violates the frozen privacy boundary")
    report_path.write_text(serialized, encoding="utf-8")
    tracker.client.set_tag(
        report["tracking"]["mlflow_parent_run_id"],
        "report_sha256",
        sha256_file(report_path),
    )
    print(
        json.dumps(
            {
                "report": report_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(report_path),
                "artifacts": report["artifacts"],
                "interpretations": report["interpretations"],
                "runtime": report["runtime"],
                "mlflow_parent_run_id": report["tracking"]["mlflow_parent_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
