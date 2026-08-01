"""Small, testable training primitives shared by E0 and E1."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_labels(dataset: Any) -> tuple[int, ...]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError("Patient dataset must expose one label per patient")
    result = tuple(int(value) for value in labels)
    if len(result) != len(dataset):
        raise ValueError("Patient labels do not match dataset length")
    return result


def make_patient_balanced_sampler(
    dataset: Any,
    seed: int,
) -> WeightedRandomSampler:
    labels = _dataset_labels(dataset)
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float = float("-inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")

    def update(self, epoch: int, score: float) -> tuple[bool, bool]:
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved, self.epochs_without_improvement >= self.patience


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> Iterable[torch.Tensor]:
    for group in optimizer.param_groups:
        yield from group["params"]


def run_patient_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int = 1,
    amp: bool = True,
    gradient_clip: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    instance_chunk_size: int | None = None,
) -> dict[str, Any]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    training = optimizer is not None
    model.train(training)
    batches = list(loader) if not hasattr(loader, "__len__") else loader
    batch_count = len(batches)
    if batch_count == 0:
        raise ValueError("Patient loader is empty")
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_patients = 0
    all_probabilities = []
    all_targets = []
    all_person_keys: list[str] = []
    all_image_counts = []
    for batch_index, batch in enumerate(batches):
        images = batch["images"].to(device, non_blocking=True)
        mask = batch["instance_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context, torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images, mask, instance_chunk_size=instance_chunk_size)
            loss = F.nll_loss(outputs["patient_log_probabilities"], targets)
        if training:
            remainder = batch_count % accumulation_steps
            in_final_partial_group = remainder and batch_index >= batch_count - remainder
            divisor = remainder if in_final_partial_group else accumulation_steps
            scaler.scale(loss / divisor).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == batch_count
            )
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(_optimizer_parameters(optimizer), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        patient_count = int(targets.shape[0])
        total_loss += float(loss.detach()) * patient_count
        total_patients += patient_count
        all_probabilities.append(outputs["patient_probabilities"].detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_person_keys.extend(batch["person_keys"])
        all_image_counts.extend(mask.sum(dim=1).detach().cpu().tolist())

    return {
        "prediction_level": "patient",
        "loss": total_loss / total_patients,
        "probabilities": torch.cat(all_probabilities),
        "targets": torch.cat(all_targets),
        "person_keys": all_person_keys,
        "image_counts": all_image_counts,
    }
