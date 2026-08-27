"""Run frozen E1 on one outer fold after replacing manual boxes with R0 OOF boxes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_auto_roi import read_csv_rows, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/r1_auto_roi_classification_sensitivity.yaml"),
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_code") != "R1":
        raise ValueError("R1 inference requires an R1 config")
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if dirty:
        raise RuntimeError("Formal R1 inference requires a clean committed worktree")

    base = config["base_experiment"]
    base_config_path = (ROOT / base["config_path"]).resolve()
    if sha256_file(base_config_path) != base["config_sha256"]:
        raise ValueError("E1 base config hash differs from the R1 preregistration")
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    checkpoint_contract = base["checkpoints"][args.fold]
    checkpoint = (ROOT / checkpoint_contract["path"]).resolve()
    if sha256_file(checkpoint) != checkpoint_contract["sha256"]:
        raise ValueError("E1 checkpoint hash differs from the R1 preregistration")

    r0_path = ROOT / config["candidate_input"]["pattern"].format(fold=args.fold)
    r0_rows = read_csv_rows(r0_path)
    overrides = {}
    for row in r0_rows:
        if int(row["outer_fold"]) != args.fold:
            raise ValueError("R0 OOF file contains the wrong outer fold")
        if row["status"] == "detected":
            overrides[row["image_key"]] = {
                axis: float(row[f"auto_roi_{axis}"])
                for axis in ("x1", "y1", "x2", "y2")
            }
        elif row["status"] == "no_detection":
            overrides[row["image_key"]] = None
        else:
            raise ValueError(f"unsupported R0 status: {row['status']}")

    from torch.utils.data import DataLoader
    import torch

    from src.research_dataset import (
        PatientBagDataset,
        ResearchImageDataset,
        collate_patient_bags,
        load_fold_records,
        replace_record_rois,
    )
    from src.research_metrics import compute_patient_metrics
    from src.research_models import MaskedMeanClassifier, create_timm_encoder
    from src.research_oof import PROBABILITY_COLUMNS
    from src.research_runtime import configure_conservative_threads, set_below_normal_priority
    from src.research_schema import DIAGNOSIS_CLASSES
    from src.research_training import run_patient_epoch, seed_everything

    configure_conservative_threads()
    set_below_normal_priority()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal R1 inference")
    seed_everything(int(config["seed"]) + args.fold)

    manual_records = load_fold_records(
        PATIENT_MULTIMODAL_REGISTRY_DIR,
        ROOT,
        outer_fold=args.fold,
        split="test",
    )
    records, fallback_keys = replace_record_rois(manual_records, overrides)
    e1_summary_path = PATIENT_MULTIMODAL_REPORTS_DIR / (
        f"E1-fold{args.fold}-seed{int(base_config['seed']) + args.fold}-formal_summary.json"
    )
    e1_summary = json.loads(e1_summary_path.read_text(encoding="utf-8"))
    fill = tuple(int(value) for value in e1_summary["letterbox_fill_rgb"])
    image_dataset = ResearchImageDataset(
        records,
        input_mode="roi",
        resize_mode=base_config["data"]["resize_mode"],
        output_size=int(base_config["data"]["output_size"]),
        image_transform=None,
        letterbox_fill=fill,
    )
    bag_dataset = PatientBagDataset(
        image_dataset,
        max_instances=int(base_config["data"]["max_instances_train"]),
        training=False,
        seed=int(config["seed"]) + args.fold,
    )
    loader = DataLoader(
        bag_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_patient_bags,
    )
    encoder, feature_dim = create_timm_encoder(
        base_config["model"]["name"], pretrained=False
    )
    model = MaskedMeanClassifier(
        encoder,
        feature_dim,
        num_classes=int(base_config["model"]["num_classes"]),
        dropout=float(base_config["model"]["dropout"]),
    ).cuda()
    checkpoint_state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint_state["model"])
    result = run_patient_epoch(
        model,
        loader,
        device=torch.device("cuda"),
        optimizer=None,
        amp=bool(base_config["training"]["amp"]),
        instance_chunk_size=int(base_config["data"]["max_instances_train"]),
    )
    metrics = compute_patient_metrics(
        result["targets"].numpy(), result["probabilities"].numpy(), DIAGNOSIS_CLASSES
    )

    prediction_path = PATIENT_MULTIMODAL_REPORTS_DIR / "oof" / f"R1_fold{args.fold}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = f"R1-E1-frozen-auto-roi-fold{args.fold}"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "prediction_level",
            "person_key",
            "outer_fold",
            "reference_class",
            "reference_id",
            *PROBABILITY_COLUMNS,
            "image_count",
            "model_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, person_key in enumerate(result["person_keys"]):
            target = int(result["targets"][index])
            probabilities = result["probabilities"][index].tolist()
            writer.writerow(
                {
                    "prediction_level": "patient",
                    "person_key": person_key,
                    "outer_fold": args.fold,
                    "reference_class": DIAGNOSIS_CLASSES[target],
                    "reference_id": target,
                    **dict(zip(PROBABILITY_COLUMNS, probabilities, strict=True)),
                    "image_count": int(result["image_counts"][index]),
                    "model_id": model_id,
                }
            )

    summary = {
        "experiment_code": "R1",
        "status": "completed_fold_inference",
        "outer_fold": args.fold,
        "classifier_retrained": False,
        "patients": len(result["person_keys"]),
        "images": len(records),
        "full_image_fallback_count": len(fallback_keys),
        "full_image_fallback_image_keys": list(fallback_keys),
        "metrics_descriptive_until_fivefold_merge": metrics,
        "checkpoint_sha256": sha256_file(checkpoint),
        "r0_oof_sha256": sha256_file(r0_path),
        "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": sha256_file(prediction_path),
        "geometry_gate": None,
        "allow_2026_labels": False,
    }
    summary_path = PATIENT_MULTIMODAL_REPORTS_DIR / f"R1-fold{args.fold}-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

