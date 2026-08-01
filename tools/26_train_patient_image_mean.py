"""Train one E0/E1 outer fold with patient-level mean probability aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_ARTIFACTS_DIR,
    PATIENT_MULTIMODAL_EXPERIMENT_DIR,
    PATIENT_MULTIMODAL_LOGS_DIR,
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_config import (  # noqa: E402
    load_research_config,
    resolve_pretrained_weights,
)
from src.research_runtime import (  # noqa: E402
    ResourcePolicy,
    collect_resource_snapshot,
    configure_conservative_threads,
    evaluate_runtime,
    evaluate_training_start,
    set_below_normal_priority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-epochs", type=int, choices=range(1, 6))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _git_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    )
    return revision, dirty


def _seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _build_loader(dataset, batch_size, workers, sampler=None, seed=0):
    import torch
    from torch.utils.data import DataLoader

    from src.research_dataset import collate_patient_bags

    generator = torch.Generator().manual_seed(seed)
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_patient_bags,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def _optimizer_and_scheduler(model, config, epochs):
    import torch

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(optimizer_config["encoder_lr"]),
            },
            {
                "params": model.head.parameters(),
                "lr": float(optimizer_config["head_lr"]),
            },
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    from src.research_training import warmup_cosine_multiplier

    warmup = int(config["training"]["warmup_epochs"])

    def multiplier(epoch: int) -> float:
        return warmup_cosine_multiplier(epoch, epochs, warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _write_predictions(path: Path, result, outer_fold: int, model_id: str) -> None:
    from src.research_oof import PROBABILITY_COLUMNS
    from src.research_schema import DIAGNOSIS_CLASSES

    probabilities = result["probabilities"].numpy()
    targets = result["targets"].numpy()
    rows = []
    for index, person_key in enumerate(result["person_keys"]):
        ranking = probabilities[index].argsort()[::-1]
        row = {
            "prediction_level": "patient",
            "person_key": person_key,
            "outer_fold": outer_fold,
            "reference_class": DIAGNOSIS_CLASSES[int(targets[index])],
            "reference_id": int(targets[index]),
            "image_count": int(result["image_counts"][index]),
            "model_id": model_id,
            "top1": DIAGNOSIS_CLASSES[int(ranking[0])],
            "top2": DIAGNOSIS_CLASSES[int(ranking[1])],
        }
        row.update(
            {
                column: float(probabilities[index, class_id])
                for class_id, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: item["person_key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_research_config(args.config.resolve())
    pretrained_path = resolve_pretrained_weights(config, ROOT)
    configure_conservative_threads()
    set_below_normal_priority()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    from src.research_dataset import (
        PatientBagDataset,
        ResearchImageDataset,
        estimate_letterbox_fill,
        load_fold_records,
    )
    from src.research_metrics import compute_patient_metrics
    from src.research_models import MaskedMeanClassifier, create_timm_encoder
    from src.research_schema import DIAGNOSIS_CLASSES
    from src.research_tracking import LocalResearchTracker
    from src.research_training import (
        EarlyStopping,
        make_patient_balanced_sampler,
        run_patient_epoch,
        seed_everything,
    )
    from src.research_transforms import build_training_augmentation

    policy = ResourcePolicy(
        soft_time_limit_hours=float(config["runtime"]["soft_limit_hours"]),
        hard_time_limit_hours=float(config["runtime"]["hard_limit_hours"]),
        gpu_memory_budget_gb=float(config["runtime"]["max_gpu_memory_gb"]),
        dataloader_workers=int(config["data"]["num_workers"]),
    )
    start_snapshot, gpu = collect_resource_snapshot(ROOT)
    start_decision = evaluate_training_start(start_snapshot, policy)
    revision, dirty = _git_revision()
    seed = int(config["seed"]) + args.fold
    seed_everything(seed)
    source_freeze = json.loads(
        (PATIENT_MULTIMODAL_REPORTS_DIR / "source_freeze.json").read_text(
            encoding="utf-8"
        )
    )

    record_sets = {
        split: load_fold_records(
            PATIENT_MULTIMODAL_REGISTRY_DIR,
            ROOT,
            outer_fold=args.fold,
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    split_counts = {
        split: {
            "patients": len({record.person_key for record in records}),
            "images": len(records),
        }
        for split, records in record_sets.items()
    }
    run_contract = {
        "experiment_code": config["experiment_code"],
        "input_mode": config["input_mode"],
        "outer_fold": args.fold,
        "seed": seed,
        "pilot": args.pilot,
        "dataset_version": source_freeze["dataset_version_short"],
        "git_revision": revision,
        "git_dirty": dirty,
        "pretrained_sha256": config["model"]["pretrained_sha256"],
        "split_counts": split_counts,
        "outer_test_used_for_training_or_early_stopping": False,
        "resource_start": asdict(start_snapshot),
        "resource_start_decision": asdict(start_decision),
        "gpu": gpu,
    }
    print(json.dumps(run_contract, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if not start_decision.allowed:
        print("Training start rejected by resource policy", file=sys.stderr)
        return 2
    if not args.pilot and dirty:
        print("Formal training requires a clean committed worktree", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA is required for a formal research run", file=sys.stderr)
        return 2

    data_config = config["data"]
    letterbox_fill = estimate_letterbox_fill(
        record_sets["train"],
        input_mode=config["input_mode"],
        seed=seed,
    )
    run_contract["letterbox_fill_rgb"] = list(letterbox_fill)
    image_datasets = {
        split: ResearchImageDataset(
            records,
            input_mode=config["input_mode"],
            output_size=int(data_config["output_size"]),
            image_transform=(
                build_training_augmentation(letterbox_fill)
                if split == "train"
                else None
            ),
            letterbox_fill=letterbox_fill,
        )
        for split, records in record_sets.items()
    }
    bag_datasets = {
        split: PatientBagDataset(
            images,
            max_instances=int(data_config["max_instances_train"]),
            training=split == "train",
            seed=seed,
        )
        for split, images in image_datasets.items()
    }
    sampler = make_patient_balanced_sampler(bag_datasets["train"], seed=seed)
    workers = int(data_config["num_workers"])
    batch_size = int(data_config["patient_batch_size"])
    loaders = {
        "train": _build_loader(
            bag_datasets["train"], batch_size, workers, sampler=sampler, seed=seed
        ),
        "validation": _build_loader(
            bag_datasets["validation"], batch_size, workers, seed=seed + 100
        ),
        # Constructing this loader is safe; iteration happens only after the
        # best epoch is fixed, and never during a resource pilot.
        "test": _build_loader(
            bag_datasets["test"], batch_size, workers, seed=seed + 200
        ),
    }

    encoder, feature_dim = create_timm_encoder(
        config["model"]["name"],
        pretrained=bool(config["model"]["pretrained"]),
        pretrained_path=pretrained_path,
    )
    model = MaskedMeanClassifier(
        encoder,
        feature_dim,
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
    ).cuda()
    epochs = (
        args.pilot_epochs
        or int(config["training"]["pilot_epochs"])
        if args.pilot
        else int(config["training"]["max_epochs"])
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, config, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"]))
    accumulation = math.ceil(
        int(data_config["effective_patient_batch_size"]) / batch_size
    )
    stopping = EarlyStopping(
        patience=int(config["training"]["early_stopping_patience"])
    )

    code = config["experiment_code"]
    mode_suffix = "pilot" if args.pilot else "formal"
    run_id = f"{code}-fold{args.fold}-seed{seed}-{mode_suffix}"
    artifact_dir = PATIENT_MULTIMODAL_ARTIFACTS_DIR / code / f"fold_{args.fold}"
    log_dir = PATIENT_MULTIMODAL_LOGS_DIR / code
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / f"best_{mode_suffix}.pt"
    resume_path = artifact_dir / f"resume_{mode_suffix}.pt"
    history_path = log_dir / f"fold_{args.fold}_{mode_suffix}_history.json"
    history = []
    first_epoch = 0
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        state = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stopping = EarlyStopping(**state["early_stopping"])
        history = state["history"]
        first_epoch = int(state["epoch"]) + 1

    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking",
        experiment_name="patient-primary-diagnosis",
    )
    started = time.monotonic()
    stop_status = "COMPLETED"
    torch.cuda.reset_peak_memory_stats()
    with tracker.parent_run(
        run_id,
        {
            "experiment_code": code,
            "prediction_level": "patient",
            "input_mode": config["input_mode"],
            "git_revision": revision,
            "git_dirty": dirty,
            "pilot": args.pilot,
        },
    ):
        with tracker.fold_run(
            f"{run_id}-child",
            {
                "outer_fold": args.fold,
                "seed": seed,
                "backbone": config["model"]["name"],
                "patients_train": split_counts["train"]["patients"],
                "patients_validation": split_counts["validation"]["patients"],
                "patients_test": split_counts["test"]["patients"],
            },
        ):
            for epoch in range(first_epoch, epochs):
                bag_datasets["train"].set_epoch(epoch)
                sampler.generator.manual_seed(seed + epoch)
                train_result = run_patient_epoch(
                    model,
                    loaders["train"],
                    device=torch.device("cuda"),
                    optimizer=optimizer,
                    accumulation_steps=accumulation,
                    amp=bool(config["training"]["amp"]),
                    gradient_clip=float(config["optimizer"]["gradient_clip"]),
                    scaler=scaler,
                )
                validation_result = run_patient_epoch(
                    model,
                    loaders["validation"],
                    device=torch.device("cuda"),
                    optimizer=None,
                    amp=bool(config["training"]["amp"]),
                    instance_chunk_size=int(data_config["max_instances_train"]),
                )
                validation_metrics = compute_patient_metrics(
                    validation_result["targets"].numpy(),
                    validation_result["probabilities"].numpy(),
                    DIAGNOSIS_CLASSES,
                )
                improved, should_stop = stopping.update(
                    epoch,
                    validation_metrics["macro_f1"],
                )
                scheduler.step()
                elapsed_hours = (time.monotonic() - started) / 3600
                snapshot, _ = collect_resource_snapshot(ROOT)
                snapshot = replace(snapshot, elapsed_hours=elapsed_hours)
                peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
                row = {
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "validation_loss": validation_result["loss"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "encoder_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                    "elapsed_hours": elapsed_hours,
                    "peak_gpu_memory_gb": peak_gpu_gb,
                    "resource": asdict(snapshot),
                }
                history.append(row)
                tracker.log_metrics(
                    {
                        "train_loss": row["train_loss"],
                        "validation_loss": row["validation_loss"],
                        "validation_macro_f1": row["validation_macro_f1"],
                        "peak_gpu_memory_gb": peak_gpu_gb,
                        "elapsed_hours": elapsed_hours,
                    },
                    step=epoch,
                )
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "early_stopping": asdict(stopping),
                    "history": history,
                    "config": config,
                }
                torch.save(state, resume_path)
                if improved:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "validation_macro_f1": stopping.best_score,
                            "config": config,
                            "model_id": run_id,
                        },
                        best_path,
                    )
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"epoch={epoch + 1}/{epochs} train_loss={row['train_loss']:.4f} "
                    f"val_macro_f1={row['validation_macro_f1']:.4f} "
                    f"elapsed={elapsed_hours:.2f}h peak_gpu={peak_gpu_gb:.2f}GB"
                )

                decision = evaluate_runtime(snapshot, policy)
                if peak_gpu_gb > policy.gpu_memory_budget_gb:
                    stop_status = "RESOURCE_GUARD_STOPPED"
                    break
                if args.pilot and elapsed_hours >= 1.0:
                    stop_status = "TIME_BUDGET_REACHED"
                    break
                if not decision.allowed:
                    stop_status = decision.status
                    break
                if should_stop and not args.pilot:
                    stop_status = "EARLY_STOPPED"
                    break

            resource_interrupted = stop_status in {
                "TIME_BUDGET_REACHED",
                "HARD_TIME_LIMIT_REACHED",
                "RESOURCE_GUARD_STOPPED",
            }
            if args.pilot or resource_interrupted:
                summary = {
                    **run_contract,
                    "status": stop_status,
                    "epochs_completed": len(history),
                    "best_validation_macro_f1": stopping.best_score,
                    "best_epoch": stopping.best_epoch,
                    "elapsed_hours": (time.monotonic() - started) / 3600,
                    "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "outer_test_iterated": False,
                }
            elif best_path.is_file():
                best = torch.load(best_path, map_location="cuda", weights_only=False)
                model.load_state_dict(best["model"])
                test_result = run_patient_epoch(
                    model,
                    loaders["test"],
                    device=torch.device("cuda"),
                    optimizer=None,
                    amp=bool(config["training"]["amp"]),
                    instance_chunk_size=int(data_config["max_instances_train"]),
                )
                test_metrics = compute_patient_metrics(
                    test_result["targets"].numpy(),
                    test_result["probabilities"].numpy(),
                    DIAGNOSIS_CLASSES,
                )
                prediction_path = (
                    PATIENT_MULTIMODAL_REPORTS_DIR
                    / "oof"
                    / f"{code}_fold{args.fold}.csv"
                )
                _write_predictions(prediction_path, test_result, args.fold, run_id)
                summary = {
                    **run_contract,
                    "status": stop_status,
                    "epochs_completed": len(history),
                    "best_validation_macro_f1": stopping.best_score,
                    "best_epoch": stopping.best_epoch,
                    "test_metrics": test_metrics,
                    "outer_test_iterated": True,
                    "prediction_file": prediction_path.name,
                }
            else:
                raise RuntimeError("No best checkpoint was produced")

    summary_path = PATIENT_MULTIMODAL_REPORTS_DIR / f"{run_id}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if stop_status in {"COMPLETED", "EARLY_STOPPED"} and resume_path.exists():
        resume_path.unlink()
    print(f"status={stop_status}")
    print(f"summary={summary_path.resolve()}")
    return 0 if stop_status in {"COMPLETED", "EARLY_STOPPED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
