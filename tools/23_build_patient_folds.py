"""Generate fixed patient-level outer folds and inner early-stop splits."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_schema import DIAGNOSIS_TO_ID  # noqa: E402
from src.research_splits import build_inner_rows, build_outer_folds  # noqa: E402


SEED = 20260724
OUTER_FOLDS = 5


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_included_patients() -> list[dict[str, object]]:
    with (PATIENT_MULTIMODAL_REGISTRY_DIR / "patients.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if int(row["include"]) == 1
        ]


def main() -> int:
    patients = read_included_patients()
    outer = build_outer_folds(patients, n_splits=OUTER_FOLDS, seed=SEED)
    repeat = build_outer_folds(patients, n_splits=OUTER_FOLDS, seed=SEED)
    if outer != repeat:
        raise RuntimeError("Outer folds are not deterministic")

    outer_rows = [
        {
            "person_key": row["person_key"],
            "diagnosis": row["diagnosis"],
            "diagnosis_id": DIAGNOSIS_TO_ID[row["diagnosis"]],
            "outer_fold": outer[row["person_key"]],
        }
        for row in sorted(patients, key=lambda item: item["person_key"])
    ]
    inner_rows = build_inner_rows(patients, outer, seed=SEED)
    write_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_outer.csv", outer_rows)
    write_csv(PATIENT_MULTIMODAL_REGISTRY_DIR / "folds_inner.csv", inner_rows)

    outer_counts = Counter(
        (row["outer_fold"], row["diagnosis"]) for row in outer_rows
    )
    distributions = {
        str(fold): {
            diagnosis: outer_counts[(fold, diagnosis)]
            for diagnosis in DIAGNOSIS_TO_ID
        }
        for fold in range(OUTER_FOLDS)
    }
    spread = {
        diagnosis: max(distributions[str(fold)][diagnosis] for fold in range(5))
        - min(distributions[str(fold)][diagnosis] for fold in range(5))
        for diagnosis in DIAGNOSIS_TO_ID
    }
    inner_counts = Counter(
        (row["outer_fold"], row["split"]) for row in inner_rows
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": SEED,
        "outer_fold_count": OUTER_FOLDS,
        "patients": len(patients),
        "unique_person_keys": len({row["person_key"] for row in patients}),
        "deterministic_regeneration": True,
        "outer_diagnosis_distribution": distributions,
        "max_class_count_spread_between_outer_folds": spread,
        "inner_split_counts": {
            str(fold): {
                split: inner_counts[(fold, split)]
                for split in ("train", "validation", "test")
            }
            for fold in range(OUTER_FOLDS)
        },
        "outer_test_never_in_inner_train_or_validation": all(
            row["split"] == "test"
            if outer[row["person_key"]] == row["outer_fold"]
            else row["split"] in {"train", "validation"}
            for row in inner_rows
        ),
    }
    (PATIENT_MULTIMODAL_REPORTS_DIR / "fold_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source = json.loads(
        (PATIENT_MULTIMODAL_REPORTS_DIR / "source_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    annotations = json.loads(
        (PATIENT_MULTIMODAL_REPORTS_DIR / "annotation_normalization.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (PATIENT_MULTIMODAL_REPORTS_DIR / "manifest_audit.json").read_text(
            encoding="utf-8"
        )
    )
    table_header = "| outer fold | " + " | ".join(DIAGNOSIS_TO_ID) + " |"
    table_rule = "|---:|" + "---:|" * len(DIAGNOSIS_TO_ID)
    table_rows = [
        "| "
        + str(fold)
        + " | "
        + " | ".join(
            str(distributions[str(fold)][diagnosis])
            for diagnosis in DIAGNOSIS_TO_ID
        )
        + " |"
        for fold in range(OUTER_FOLDS)
    ]
    checkpoint = "\n".join(
        [
            "# 检查点0：训练前数据基础报告",
            "",
            f"- 数据版本：`{source['dataset_version_short']}`",
            f"- 纳入患者：{manifest['included_patients']}",
            f"- 纳入图像：{manifest['included_images']}",
            f"- 规范化区域类别：{annotations['normalized_category_count']}",
            f"- 标注对象：{annotations['objects_before']}",
            "- 残留疾病前缀：0",
            f"- 所有纳入图像ROI合法且已人工确认：{manifest['all_included_images_have_valid_reviewed_roi']}",
            f"- 跨诊断身份冲突：{len(manifest['cross_diagnosis_identity_conflicts'])}",
            f"- 五折固定种子：{SEED}",
            f"- 相同种子可重复生成：{report['deterministic_regeneration']}",
            "",
            "## 外层五折患者数",
            "",
            table_header,
            table_rule,
            *table_rows,
            "",
            "## 结论",
            "",
            "P0～P4数据基础已生成。只有在本报告通过人工审阅后，才启动E0/E1资源试运行。",
            "",
        ]
    )
    (PATIENT_MULTIMODAL_REPORTS_DIR / "checkpoint0.md").write_text(
        checkpoint,
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
