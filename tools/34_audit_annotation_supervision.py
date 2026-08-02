"""Audit normalized polygon supervision without exporting source identities."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

# Keep this CPU audit responsive on the shared workstation. These variables must
# be set before NumPy/scikit-learn are imported through the project modules.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_DERIVED_DIR,
    PATIENT_MULTIMODAL_EXPERIMENT_DIR,
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_annotation_audit import (  # noqa: E402
    audit_annotation_records,
    evaluate_manual_presence_proxy,
)
from src.research_ledger import sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402
from src.research_tracking import LocalResearchTracker  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/annotation_supervision_audit.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    below_normal_priority = set_below_normal_priority()
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    annotation_root = (
        PATIENT_MULTIMODAL_DERIVED_DIR / str(config["annotation_version"])
    )
    mapping_path = annotation_root / "category_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    categories = list(mapping["categories"])
    if len(categories) != int(config["expected_categories"]):
        raise ValueError("Unexpected normalized category count")
    roles = dict(config["category_roles"])
    if set(roles) != set(categories):
        raise ValueError("Configured roles do not match category mapping")

    images = _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv")
    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_outer.csv")
    }
    included_images = []
    for row in images:
        if int(row["include"]) != 1:
            continue
        included_images.append({**row, "outer_fold": folds[row["person_key"]]})

    sources = {
        row["image_key"]: row.get("normalized_annotation_path", "")
        for row in _read_csv(
            PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
        )
    }
    annotations = {}
    for row in included_images:
        relative = sources.get(row["image_key"], "")
        if not relative:
            continue
        path = ROOT / relative
        if path.is_file():
            annotations[row["image_key"]] = json.loads(
                path.read_text(encoding="utf-8")
            )

    report, patient_labels = audit_annotation_records(
        included_images,
        annotations,
        categories,
        roles,
        config["support_thresholds"],
    )
    report["provenance"] = {
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
        "config_sha256": sha256_file(config_path),
        "category_mapping_sha256": sha256_file(mapping_path),
        "private_linkage_sha256": sha256_file(
            PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
        ),
    }
    if report["coverage"].get("unknown_category_objects", 0):
        raise ValueError("Normalized annotations contain an unknown category")
    if report["coverage"].get("residual_disease_prefix_objects", 0):
        raise ValueError("Normalized annotations retain a disease-prefixed category")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "contract": report["contract"],
                    "coverage": report["coverage"],
                    "support_summary": report["support_summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    patient_rows_by_key = {}
    for row in included_images:
        patient_rows_by_key[row["person_key"]] = {
            "person_key": row["person_key"],
            "diagnosis": row["diagnosis"],
            "diagnosis_id": int(row["diagnosis_id"]),
            "outer_fold": int(row["outer_fold"]),
        }
    proxy, oof_rows = evaluate_manual_presence_proxy(
        list(patient_rows_by_key.values()),
        patient_labels,
        categories,
        roles,
    )
    report["manual_annotation_presence_proxy"] = proxy
    report["runtime"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numeric_thread_limit": 2,
        "below_normal_priority": below_normal_priority,
        "gpu_used": False,
    }

    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking",
        "patient-primary-diagnosis",
    )
    tracking_metadata = {
        "study": config["study"],
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
        "config_sha256": report["provenance"]["config_sha256"],
        "folds": "0,1,2,3,4",
        "seeds": ",".join(
            str(seed) for seed in config["manual_presence_proxy"]["seeds"]
        ),
        "status": "completed",
    }
    with tracker.parent_run(
        "annotation-supervision-audit-formal", tracking_metadata
    ) as run:
        mlflow_run_id = run.info.run_id
        tracker.log_metrics(
            {
                f"proxy_{group}_macro_f1": values["metrics"]["macro_f1"]
                for group, values in proxy.items()
            }
        )
        tracker.log_metrics(
            {
                f"proxy_{group}_macro_auc": values["metrics"]["macro_auc"]
                for group, values in proxy.items()
            }
        )
    report["tracking"] = {
        "mlflow_experiment": "patient-primary-diagnosis",
        "mlflow_parent_run_id": mlflow_run_id,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    forbidden = ("raw_image_path", "raw_annotation_path", "normalized_annotation_path")
    if any(token in serialized for token in forbidden):
        raise ValueError("Audit report contains a source path field")

    output_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_supervision"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "annotation_supervision_audit.json"
    report_path.write_text(serialized, encoding="utf-8")
    category_path = output_dir / "annotation_category_support.csv"
    category_rows = report["categories"]
    with category_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "category",
            "role",
            "objects",
            "images",
            "patients",
            "fold_patient_counts",
            "top_diagnosis",
            "top_diagnosis_patient_share",
            "support_tier",
            "median_polygon_to_roi_area",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in category_rows:
            writer.writerow({key: row[key] for key in fields})
    oof_path = output_dir / "annotation_presence_proxy_oof.csv"
    with oof_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_rows[0]))
        writer.writeheader()
        writer.writerows(oof_rows)

    print(
        json.dumps(
            {
                "report": report_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(report_path),
                "category_support_sha256": sha256_file(category_path),
                "proxy_oof_sha256": sha256_file(oof_path),
                "mlflow_parent_run_id": mlflow_run_id,
                "support_summary": report["support_summary"],
                "proxy_macro_f1": {
                    group: values["metrics"]["macro_f1"]
                    for group, values in proxy.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
