"""Train one patient-image outer fold with mean or gated-attention aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
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
    RuntimeGuard,
    collect_resource_snapshot,
    configure_conservative_threads,
    evaluate_training_start,
    set_below_normal_priority,
)
from src.research_ledger import sha256_file  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-epochs", type=int, choices=range(1, 6))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watchdog-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _load_patient_image_config(path: Path) -> dict:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("experiment_code") == "G0":
        from src.research_gate import load_gate_config

        return load_gate_config(path)
    return load_research_config(path)


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
    aggregation_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    if not aggregation_parameters:
        raise ValueError("Patient model has no aggregation/head parameters")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(optimizer_config["encoder_lr"]),
            },
            {
                "params": aggregation_parameters,
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


def _write_attention(
    path: Path,
    result,
    outer_fold: int,
    model_id: str,
) -> None:
    summaries = result.get("attention_summaries", [])
    if not summaries:
        raise ValueError("MIL test result has no attention summaries")
    rows = []
    for summary in summaries:
        image_keys = summary["image_keys"]
        weights = summary["attention_weights"]
        if len(image_keys) != len(weights):
            raise ValueError("MIL image keys and attention weights do not match")
        ranking = sorted(range(len(weights)), key=lambda index: weights[index], reverse=True)
        ranks = {image_index: rank + 1 for rank, image_index in enumerate(ranking)}
        for image_index, (image_key, weight) in enumerate(zip(image_keys, weights)):
            rows.append(
                {
                    "prediction_level": "image_importance",
                    "person_key": summary["person_key"],
                    "image_key": image_key,
                    "outer_fold": outer_fold,
                    "model_id": model_id,
                    "image_count": len(image_keys),
                    "attention_weight": float(weight),
                    "attention_rank": ranks[image_index],
                }
            )
    rows.sort(key=lambda item: (item["person_key"], item["attention_rank"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)
    config_path = args.config.resolve()
    config = _load_patient_image_config(config_path)
    pretrained_path = resolve_pretrained_weights(config, ROOT)
    if not args.dry_run and not args.watchdog_child:
        from src.research_watchdog import run_with_hard_timeout

        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *raw_args,
            "--watchdog-child",
        ]
        watched = run_with_hard_timeout(
            command,
            cwd=ROOT,
            timeout_seconds=float(config["runtime"]["hard_limit_hours"]) * 3600,
        )
        if watched.timed_out:
            destination = (
                PATIENT_MULTIMODAL_REPORTS_DIR
                / f"{config['experiment_code']}-fold{args.fold}-hard-timeout.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "status": "HARD_TIME_LIMIT_REACHED",
                        "outer_fold": args.fold,
                        "experiment_code": config["experiment_code"],
                        "hard_limit_hours": config["runtime"]["hard_limit_hours"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return watched.returncode
    configure_conservative_threads()
    set_below_normal_priority()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(int(config["runtime"].get("max_cpu_threads", 4)))
    torch.set_num_interop_threads(
        int(config["runtime"].get("max_interop_threads", 1))
    )

    from src.research_dataset import (
        PatientBagDataset,
        ResearchImageDataset,
        estimate_letterbox_fill,
        load_fold_records,
    )
    from src.research_metrics import compute_patient_metrics
    from src.research_mil import GatedAttentionMILClassifier, summarize_attention
    from src.research_models import (
        MaskedMeanClassifier,
        MaskedMeanFeatureClassifier,
        create_timm_encoder,
    )
    from src.research_schema import DIAGNOSIS_CLASSES
    from src.research_tracking import LocalResearchTracker
    from src.research_training import (
        EarlyStopping,
        make_patient_balanced_sampler,
        previous_elapsed_hours,
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
    is_gate = config["experiment_code"] == "G0"
    if is_gate and source_freeze["dataset_version"] != config["data_fingerprint"]:
        raise ValueError("G0 config and frozen dataset fingerprint do not match")

    record_sets = {
        split: load_fold_records(
            PATIENT_MULTIMODAL_REGISTRY_DIR,
            ROOT,
            outer_fold=args.fold,
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    if is_gate:
        from src.research_gate import remap_records_to_gate

        record_sets = {
            split: remap_records_to_gate(records)
            for split, records in record_sets.items()
        }
    split_counts = {
        split: {
            "patients": len({record.person_key for record in records}),
            "images": len(records),
        }
        for split, records in record_sets.items()
    }
    if is_gate:
        gate_people = {}
        for records in record_sets.values():
            for record in records:
                previous = gate_people.setdefault(record.person_key, record.diagnosis_id)
                if previous != record.diagnosis_id:
                    raise ValueError("G0 patient has mixed binary labels across splits")
        normal_patients = sum(label == 0 for label in gate_people.values())
        abnormal_patients = sum(label == 1 for label in gate_people.values())
        expected = config["data"]
        if len(gate_people) != int(expected["expected_patients"]):
            raise ValueError("G0 patient count differs from frozen config")
        if sum(item["images"] for item in split_counts.values()) != int(
            expected["expected_images"]
        ):
            raise ValueError("G0 image count differs from frozen config")
        if normal_patients != int(expected["expected_normal_patients"]):
            raise ValueError("G0 normal patient count differs from frozen config")
        if abnormal_patients != int(expected["expected_abnormal_patients"]):
            raise ValueError("G0 abnormal patient count differs from frozen config")
        gate_counts = {
            "patients": len(gate_people),
            "images": sum(item["images"] for item in split_counts.values()),
            "normal_patients": normal_patients,
            "abnormal_patients": abnormal_patients,
        }
    else:
        gate_counts = None
    run_contract = {
        "experiment_code": config["experiment_code"],
        "input_mode": config["input_mode"],
        "resize_mode": config["data"]["resize_mode"],
        "aggregation": config["model"].get("aggregation", "mean_probability"),
        "attention_kl_weight": float(
            config["training"].get("attention_kl_weight", 0.0)
        ),
        "outer_fold": args.fold,
        "seed": seed,
        "pilot": args.pilot,
        "dataset_version": source_freeze["dataset_version_short"],
        "git_revision": revision,
        "git_dirty": dirty,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "pretrained_sha256": config["model"]["pretrained_sha256"],
        "split_counts": split_counts,
        "outer_test_used_for_training_or_early_stopping": False,
        "resource_start": asdict(start_snapshot),
        "resource_start_decision": asdict(start_decision),
        "gpu": gpu,
        "mlflow_database": (
            PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking" / "mlflow.db"
        ).relative_to(ROOT).as_posix(),
    }
    if is_gate:
        run_contract.update(
            {
                "task_type": config["task"]["type"],
                "data_fingerprint": config["data_fingerprint"],
                "gate_counts": gate_counts,
            }
        )
    print(json.dumps(run_contract, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if is_gate:
        print("G0 formal training path is not enabled in this increment", file=sys.stderr)
        return 2
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
    resize_mode = data_config["resize_mode"]
    letterbox_fill = estimate_letterbox_fill(
        record_sets["train"],
        input_mode=config["input_mode"],
        seed=seed,
    )
    run_contract["augmentation_fill_rgb"] = list(letterbox_fill)
    if resize_mode == "letterbox":
        run_contract["letterbox_fill_rgb"] = list(letterbox_fill)
    image_datasets = {
        split: ResearchImageDataset(
            records,
            input_mode=config["input_mode"],
            resize_mode=resize_mode,
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
    aggregation = config["model"].get("aggregation", "mean_probability")
    if aggregation == "gated_attention":
        model = GatedAttentionMILClassifier(
            encoder,
            feature_dim,
            num_classes=int(config["model"]["num_classes"]),
            attention_dim=int(config["model"]["attention_dim"]),
            dropout=float(config["model"]["dropout"]),
        ).cuda()
    elif aggregation == "mean_feature":
        model = MaskedMeanFeatureClassifier(
            encoder,
            feature_dim,
            num_classes=int(config["model"]["num_classes"]),
            dropout=float(config["model"]["dropout"]),
        ).cuda()
    else:
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
    runtime_guard = RuntimeGuard(policy)

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
    prior_elapsed_hours = 0.0
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
        prior_elapsed_hours = previous_elapsed_hours(history)
    run_contract["resume_requested"] = bool(args.resume)
    run_contract["resume_from_epoch"] = first_epoch if args.resume else None
    run_contract["prior_elapsed_hours"] = prior_elapsed_hours

    tracker = LocalResearchTracker(
        PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking",
        experiment_name="patient-primary-diagnosis",
    )
    started = time.monotonic()
    stop_status = "COMPLETED"
    stop_reasons: list[str] = []
    torch.cuda.reset_peak_memory_stats()
    with tracker.parent_run(
        run_id,
        {
            "experiment_code": code,
            "prediction_level": "patient",
            "input_mode": config["input_mode"],
            "resize_mode": resize_mode,
            "aggregation": aggregation,
            "attention_kl_weight": float(
                config["training"].get("attention_kl_weight", 0.0)
            ),
            "git_revision": revision,
            "git_dirty": dirty,
            "pilot": args.pilot,
            "resume_requested": bool(args.resume),
        },
    ) as parent_run:
        run_contract["mlflow_parent_run_id"] = parent_run.info.run_id
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
        ) as fold_run:
            run_contract["mlflow_fold_run_id"] = fold_run.info.run_id
            for epoch in range(first_epoch, epochs):
                bag_datasets["train"].set_epoch(epoch)
                sampler.generator.manual_seed(seed + epoch)
                encoder_lr_used = optimizer.param_groups[0]["lr"]
                head_lr_used = optimizer.param_groups[1]["lr"]
                train_result = run_patient_epoch(
                    model,
                    loaders["train"],
                    device=torch.device("cuda"),
                    optimizer=optimizer,
                    accumulation_steps=accumulation,
                    amp=bool(config["training"]["amp"]),
                    gradient_clip=float(config["optimizer"]["gradient_clip"]),
                    scaler=scaler,
                    attention_kl_weight=float(
                        config["training"].get("attention_kl_weight", 0.0)
                    ),
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
                attempt_elapsed_hours = (time.monotonic() - started) / 3600
                elapsed_hours = prior_elapsed_hours + attempt_elapsed_hours
                snapshot, _ = collect_resource_snapshot(ROOT)
                snapshot = replace(snapshot, elapsed_hours=elapsed_hours)
                peak_allocated_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
                peak_reserved_gpu_gb = torch.cuda.max_memory_reserved() / (1024**3)
                row = {
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "train_classification_loss": train_result[
                        "classification_loss"
                    ],
                    "train_attention_regularization": train_result[
                        "attention_regularization"
                    ],
                    "validation_loss": validation_result["loss"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "encoder_lr": encoder_lr_used,
                    "head_lr": head_lr_used,
                    "elapsed_hours": elapsed_hours,
                    "attempt_elapsed_hours": attempt_elapsed_hours,
                    "elapsed_hours_total": elapsed_hours,
                    "peak_gpu_memory_allocated_gb": peak_allocated_gpu_gb,
                    "peak_gpu_memory_reserved_gb": peak_reserved_gpu_gb,
                    "resource": asdict(snapshot),
                }
                history.append(row)
                tracker.log_metrics(
                    {
                        "train_loss": row["train_loss"],
                        "train_classification_loss": row[
                            "train_classification_loss"
                        ],
                        "train_attention_regularization": row[
                            "train_attention_regularization"
                        ],
                        "validation_loss": row["validation_loss"],
                        "validation_macro_f1": row["validation_macro_f1"],
                        "peak_gpu_memory_allocated_gb": peak_allocated_gpu_gb,
                        "peak_gpu_memory_reserved_gb": peak_reserved_gpu_gb,
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
                    f"elapsed={elapsed_hours:.2f}h "
                    f"peak_gpu_reserved={peak_reserved_gpu_gb:.2f}GB"
                )

                decision = runtime_guard.evaluate(snapshot)
                if peak_reserved_gpu_gb > policy.gpu_memory_budget_gb:
                    stop_status = "RESOURCE_GUARD_STOPPED"
                    stop_reasons = ["peak_gpu_memory_above_configured_budget"]
                    break
                if args.pilot and elapsed_hours >= 1.0:
                    stop_status = "TIME_BUDGET_REACHED"
                    stop_reasons = ["pilot_elapsed_above_1h"]
                    break
                if not decision.allowed:
                    stop_status = decision.status
                    stop_reasons = list(decision.reasons)
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
                    "elapsed_hours": (
                        prior_elapsed_hours
                        + (time.monotonic() - started) / 3600
                    ),
                    "attempt_elapsed_hours": (time.monotonic() - started) / 3600,
                    "elapsed_hours_total": (
                        prior_elapsed_hours
                        + (time.monotonic() - started) / 3600
                    ),
                    "peak_gpu_memory_allocated_gb": (
                        torch.cuda.max_memory_allocated() / (1024**3)
                    ),
                    "peak_gpu_memory_reserved_gb": (
                        torch.cuda.max_memory_reserved() / (1024**3)
                    ),
                    "outer_test_iterated": False,
                    "stop_reason": stop_status,
                    "stop_reasons": stop_reasons,
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
                    collect_attention=aggregation == "gated_attention",
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
                prediction_relative = prediction_path.relative_to(ROOT).as_posix()
                summary = {
                    **run_contract,
                    "status": stop_status,
                    "epochs_completed": len(history),
                    "best_validation_macro_f1": stopping.best_score,
                    "best_epoch": stopping.best_epoch,
                    "test_metrics": test_metrics,
                    "outer_test_iterated": True,
                    "prediction_file": prediction_path.name,
                    "prediction_path": prediction_relative,
                    "prediction_sha256": sha256_file(prediction_path),
                    "elapsed_hours": (
                        prior_elapsed_hours
                        + (time.monotonic() - started) / 3600
                    ),
                    "attempt_elapsed_hours": (time.monotonic() - started) / 3600,
                    "elapsed_hours_total": (
                        prior_elapsed_hours
                        + (time.monotonic() - started) / 3600
                    ),
                    "peak_gpu_memory_allocated_gb": (
                        torch.cuda.max_memory_allocated() / (1024**3)
                    ),
                    "peak_gpu_memory_reserved_gb": (
                        torch.cuda.max_memory_reserved() / (1024**3)
                    ),
                    "stop_reason": stop_status,
                    "stop_reasons": stop_reasons,
                }
                if aggregation == "gated_attention":
                    attention_path = (
                        PATIENT_MULTIMODAL_REPORTS_DIR
                        / "attention"
                        / f"{code}_fold{args.fold}.csv"
                    )
                    _write_attention(
                        attention_path,
                        test_result,
                        args.fold,
                        run_id,
                    )
                    attention_relative = attention_path.relative_to(ROOT).as_posix()
                    attention_audit = summarize_attention(
                        test_result["attention_summaries"],
                        collapse_threshold=float(
                            config["model"]["attention_collapse_threshold"]
                        ),
                    )
                    attention_audit["max_allowed_multi_image_collapse_rate"] = float(
                        config["model"]["max_multi_image_collapse_rate"]
                    )
                    attention_audit["collapse_gate_passed"] = bool(
                        attention_audit["multi_image_collapse_rate"]
                        <= attention_audit["max_allowed_multi_image_collapse_rate"]
                    )
                    summary.update(
                        {
                            "attention_path": attention_relative,
                            "attention_sha256": sha256_file(attention_path),
                            "attention_audit": attention_audit,
                        }
                    )
            else:
                raise RuntimeError("No best checkpoint was produced")

    if best_path.is_file():
        summary["best_checkpoint_path"] = best_path.relative_to(ROOT).as_posix()
        summary["best_checkpoint_sha256"] = sha256_file(best_path)
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
