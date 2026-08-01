"""Audit whether acquisition/export proxies alone predict the six diagnoses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_proxy_audit import (  # noqa: E402
    FEATURE_GROUPS,
    aggregate_patient_proxy_features,
    assess_proxy_risk,
    extract_image_proxy_features,
    proxy_permutation_test,
    run_proxy_oof,
)
from src.research_runtime import (  # noqa: E402
    ResourcePolicy,
    configure_conservative_threads,
    set_below_normal_priority,
)
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATIENT_MULTIMODAL_REPORTS_DIR / "proxy_audit",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_project_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Raw image path escapes the project root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _registry_digest(registry_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "images.csv",
        "patients.csv",
        "folds_outer.csv",
        "private/image_sources.csv",
        "reference_standard.json",
    ):
        path = registry_dir / relative
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _extract_rows(registry_dir: Path) -> list[dict[str, float | int | str]]:
    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in _read_csv(registry_dir / "folds_outer.csv")
    }
    sources = {
        row["image_key"]: row["raw_image_path"]
        for row in _read_csv(registry_dir / "private" / "image_sources.csv")
    }
    images = [
        row
        for row in _read_csv(registry_dir / "images.csv")
        if row["include"] == "1"
    ]
    rows: list[dict[str, float | int | str]] = []
    started = time.monotonic()
    for index, image_row in enumerate(images, start=1):
        image_key = image_row["image_key"]
        person_key = image_row["person_key"]
        if image_key not in sources or person_key not in folds:
            raise ValueError(f"Incomplete registry link for {image_key}")
        path = _safe_project_path(sources[image_key])
        features = extract_image_proxy_features(
            path,
            roi={
                name: float(image_row[f"roi_{name}"])
                for name in ("x1", "y1", "x2", "y2")
            },
        )
        if (
            int(features["width"]) != int(image_row["width"])
            or int(features["height"]) != int(image_row["height"])
        ):
            raise ValueError(f"Registry dimensions changed for {image_key}")
        rows.append(
            {
                "image_key": image_key,
                "person_key": person_key,
                "diagnosis_id": int(image_row["diagnosis_id"]),
                "outer_fold": folds[person_key],
                **features,
            }
        )
        if index % 250 == 0 or index == len(images):
            elapsed = time.monotonic() - started
            print(f"features {index}/{len(images)} elapsed={elapsed:.1f}s", flush=True)
    if len({str(row["image_key"]) for row in rows}) != len(rows):
        raise ValueError("Image proxy feature rows are not unique")
    return rows


def _write_feature_cache(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_oof(path: Path, table, probabilities: np.ndarray, feature_group: str) -> None:
    from src.research_oof import PROBABILITY_COLUMNS

    rows = []
    for index, person_key in enumerate(table.person_keys):
        row = {
            "prediction_level": "patient",
            "person_key": person_key,
            "outer_fold": int(table.outer_folds[index]),
            "reference_class": DIAGNOSIS_CLASSES[int(table.targets[index])],
            "reference_id": int(table.targets[index]),
            "image_count": int(table.features[index, 0]),
            "model_id": f"proxy-{feature_group}",
        }
        row.update(
            {
                column: float(probabilities[index, class_id])
                for class_id, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.permutations < 1 or args.bootstrap < 1:
        raise ValueError("Permutation and bootstrap counts must be positive")
    configure_conservative_threads(ResourcePolicy())
    priority_reduced = set_below_normal_priority()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_dir = PATIENT_MULTIMODAL_REGISTRY_DIR.resolve()

    rows = _extract_rows(registry_dir)
    feature_cache = output_dir / "image_proxy_features.csv"
    _write_feature_cache(feature_cache, rows)

    from src.research_metrics import (
        bootstrap_macro_f1_ci,
        compute_patient_metrics,
        validate_oof_predictions,
    )

    expected_people = {row["person_key"] for row in rows}
    results = {}
    for group_index, (group_name, image_features) in enumerate(FEATURE_GROUPS.items()):
        print(f"evaluating {group_name}", flush=True)
        table = aggregate_patient_proxy_features(rows, image_features)
        probabilities = run_proxy_oof(
            table,
            class_count=len(DIAGNOSIS_CLASSES),
            seed=args.seed,
        )
        validate_oof_predictions(
            table.person_keys,
            table.targets,
            probabilities,
            expected_person_keys=expected_people,
            prediction_level="patient",
        )
        metrics = compute_patient_metrics(
            table.targets,
            probabilities,
            DIAGNOSIS_CLASSES,
        )
        ci = bootstrap_macro_f1_ci(
            table.targets,
            probabilities,
            args.bootstrap,
            seed=args.seed,
        )
        permutation = proxy_permutation_test(
            table,
            class_count=len(DIAGNOSIS_CLASSES),
            observed_macro_f1=float(metrics["macro_f1"]),
            count=args.permutations,
            seed=args.seed + group_index * 100_000,
        )
        risk = assess_proxy_risk(
            float(metrics["macro_f1"]),
            float(metrics["macro_auc"]),
            float(permutation["p_value"]),
        )
        _write_oof(
            output_dir / f"{group_name}_oof.csv",
            table,
            probabilities,
            group_name,
        )
        results[group_name] = {
            "risk": risk,
            "image_feature_count": len(image_features),
            "patient_feature_count": len(table.feature_names),
            "image_features": list(image_features),
            "metrics": metrics,
            "macro_f1_95_ci": ci,
            "permutation": permutation,
        }
        print(
            f"{group_name}: macro_f1={metrics['macro_f1']:.4f} "
            f"macro_auc={metrics['macro_auc']:.4f} "
            f"p={permutation['p_value']:.4f} risk={risk}",
            flush=True,
        )

    report = {
        "audit": "nonmedical_acquisition_export_proxy",
        "prediction_level": "patient",
        "patients": len(expected_people),
        "images": len(rows),
        "classes": list(DIAGNOSIS_CLASSES),
        "outer_folds": sorted({int(row["outer_fold"]) for row in rows}),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap,
        "permutation_samples": args.permutations,
        "registry_sha256": _registry_digest(registry_dir),
        "priority_reduced": priority_reduced,
        "prohibited_features": [
            "diagnosis text",
            "person identity",
            "directory name",
            "file name",
            "raw path",
            "date",
        ],
        "fixed_risk_rule": {
            "high": "p<=0.05 and (macro_f1>=0.20 or macro_auc>=0.65)",
            "moderate": "p<=0.05 or macro_f1>=0.17 or macro_auc>=0.58",
            "low": "otherwise",
        },
        "results": results,
    }
    report_path = output_dir / "nonmedical_proxy_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
