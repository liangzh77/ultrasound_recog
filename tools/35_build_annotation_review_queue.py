"""Build a deterministic, blinded draft queue for clinical annotation review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_annotation_review import (  # noqa: E402
    PUBLIC_REVIEW_FIELDS,
    build_blinded_review_queue,
)
from src.research_ledger import sha256_file  # noqa: E402
from src.research_runtime import set_below_normal_priority  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research/annotation_review_queue_v0.yaml",
    )
    parser.add_argument(
        "--write-draft",
        action="store_true",
        help="Write the pseudonymous draft CSV and aggregate audit JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_below_normal_priority()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ontology_path = ROOT / str(config["ontology_config"])
    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    if config["dataset_fingerprint"] != ontology["dataset_fingerprint"]:
        raise ValueError("Review queue and ontology data fingerprints differ")
    targets = list(config["review_targets"])
    if not set(targets) <= set(ontology["categories"]):
        raise ValueError("Review target is absent from the ontology")

    folds = {
        row["person_key"]: int(row["outer_fold"])
        for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_outer.csv")
    }
    image_rows = []
    for row in _read_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "images.csv"):
        if int(row["include"]) != 1:
            continue
        image_rows.append({**row, "outer_fold": folds[row["person_key"]]})

    source_rows = _read_csv(
        PATIENT_MULTIMODAL_REGISTRY_DIR / "private" / "image_sources.csv"
    )
    annotation_paths = {
        row["image_key"]: row.get("normalized_annotation_path", "")
        for row in source_rows
    }
    image_categories = {}
    for row in image_rows:
        relative = annotation_paths.get(row["image_key"], "")
        if not relative:
            image_categories[row["image_key"]] = set()
            continue
        annotation = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        image_categories[row["image_key"]] = {
            str(item.get("category", "")).strip()
            for item in annotation.get("objects", [])
        }

    selection = config["selection"]
    per_fold = int(selection["per_fold_per_bucket"])
    if int(selection["per_target_existing_positive"]) != per_fold * 5:
        raise ValueError("Existing-positive count must equal five fold quotas")
    if int(selection["per_target_legacy_unlabeled_candidate"]) != per_fold * 5:
        raise ValueError("Legacy-unlabeled count must equal five fold quotas")
    public_rows, audit = build_blinded_review_queue(
        image_rows,
        image_categories,
        targets,
        int(config["seed"]),
        per_fold,
        int(config["required_independent_reviews"]),
    )
    audit["provenance"] = {
        "status": config["status"],
        "dataset_fingerprint": config["dataset_fingerprint"],
        "annotation_version": config["annotation_version"],
        "config_sha256": sha256_file(config_path),
        "ontology_sha256": sha256_file(ontology_path),
    }

    if not args.write_draft:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    output_dir = PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "annotation_review_queue_draft.csv"
    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PUBLIC_REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(public_rows)
    audit["artifacts"] = {
        "queue": queue_path.relative_to(ROOT).as_posix(),
        "queue_sha256": sha256_file(queue_path),
    }
    audit_path = output_dir / "annotation_review_queue_draft_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "queue": queue_path.relative_to(ROOT).as_posix(),
                "queue_sha256": sha256_file(queue_path),
                "audit": audit_path.relative_to(ROOT).as_posix(),
                "audit_sha256": sha256_file(audit_path),
                "rows": len(public_rows),
                "unique_images": len({row["image_key"] for row in public_rows}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
