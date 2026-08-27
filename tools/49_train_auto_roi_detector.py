"""Train one patient-isolated outer fold of the R0 ultrasound-view detector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_auto_roi import load_r0_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/r0_auto_ultrasound_view_roi.yaml"),
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_r0_config((ROOT / args.config).resolve())
    data = config["data"]
    model_config = config["model"]
    runtime = config["runtime"]
    dataset_dir = (ROOT / data["output_dir"] / f"fold_{args.fold}").resolve()
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"R0 fold dataset is missing; run tools/48_prepare_auto_roi_dataset.py --fold {args.fold}"
        )

    max_threads = str(int(runtime["max_cpu_threads"]))
    os.environ.setdefault("OMP_NUM_THREADS", max_threads)
    os.environ.setdefault("MKL_NUM_THREADS", max_threads)

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(runtime["max_cpu_threads"]))
    torch.set_num_interop_threads(int(runtime["max_interop_threads"]))
    weights = (ROOT / model_config["initial_weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    mode = "pilot" if args.pilot else "formal"
    epochs = int(model_config["pilot_epochs"] if args.pilot else model_config["formal_epochs"])
    project_dir = (
        ROOT
        / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/artifacts/R0_auto_roi"
    ).resolve()
    tracking_database = (
        ROOT
        / "workspace/experiments/active/exp_2026-07_patient_multimodal_v1/tracking/mlflow.db"
    ).resolve()
    tracking_database.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_database.as_posix()}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "automatic-ultrasound-roi"

    base_run_name = f"fold_{args.fold}_{mode}"
    run_name = base_run_name
    attempt = 1
    while (project_dir / run_name).exists():
        attempt += 1
        run_name = f"{base_run_name}_attempt{attempt}"
    os.environ["MLFLOW_RUN"] = run_name
    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=int(model_config["patience"]),
        imgsz=int(model_config["image_size"]),
        batch=int(model_config["batch_size"]),
        device=runtime["device"],
        workers=int(runtime["workers"]),
        amp=bool(runtime["amp"]),
        cache=bool(runtime["cache_images"]),
        seed=int(config["seed"]) + args.fold,
        deterministic=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    trainer = model.trainer
    summary = {
        "fold": args.fold,
        "mode": mode,
        "epochs_requested": epochs,
        "outer_test_used_during_training": False,
        "run_dir": str(Path(trainer.save_dir).resolve()),
        "best_checkpoint": str(Path(trainer.best).resolve()),
        "geometry_gate": None,
        "primary_decision_basis": "downstream_patient_classification",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
