"""Patient-level image models for comparable E0/E1 experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def create_timm_encoder(
    model_name: str,
    pretrained: bool,
    pretrained_path: Path | None = None,
) -> tuple[nn.Module, int]:
    import timm

    options = {}
    if pretrained_path is not None:
        if not pretrained:
            raise ValueError("A pretrained path requires pretrained=True")
        options["pretrained_cfg_overlay"] = {
            "file": str(pretrained_path.resolve()),
            "hf_hub_id": None,
            "url": "",
        }
    encoder = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        **options,
    )
    return encoder, int(encoder.num_features)


class MaskedMeanClassifier(nn.Module):
    """Average per-image probabilities into one patient prediction."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5 or instance_mask.shape != images.shape[:2]:
            raise ValueError("Expected images [B,N,C,H,W] and mask [B,N]")
        if not instance_mask.any(dim=1).all():
            raise ValueError("Every patient bag must contain at least one image")
        valid_positions = instance_mask.nonzero(as_tuple=False)
        valid_images = images[instance_mask]
        if instance_chunk_size is None:
            features = self.encoder(valid_images)
        else:
            if instance_chunk_size < 1:
                raise ValueError("instance_chunk_size must be positive")
            features = torch.cat(
                [
                    self.encoder(valid_images[start : start + instance_chunk_size])
                    for start in range(0, len(valid_images), instance_chunk_size)
                ]
            )
        instance_logits = self.head(self.dropout(features))
        instance_probabilities = torch.softmax(instance_logits, dim=1)

        batch_size = images.shape[0]
        patient_sums = instance_probabilities.new_zeros(
            (batch_size, self.num_classes)
        )
        patient_sums.index_add_(
            0,
            valid_positions[:, 0],
            instance_probabilities,
        )
        counts = instance_mask.sum(dim=1, keepdim=True).to(
            instance_probabilities.dtype
        )
        patient_probabilities = patient_sums / counts
        return {
            "patient_probabilities": patient_probabilities,
            "patient_log_probabilities": torch.log(
                patient_probabilities.clamp_min(1e-8)
            ),
            "instance_probabilities": instance_probabilities,
            "valid_positions": valid_positions,
        }


class MaskedMeanFeatureClassifier(nn.Module):
    """Average image embeddings before one patient-level classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5 or instance_mask.shape != images.shape[:2]:
            raise ValueError("Expected images [B,N,C,H,W] and mask [B,N]")
        if not instance_mask.any(dim=1).all():
            raise ValueError("Every patient bag must contain at least one image")

        valid_positions = instance_mask.nonzero(as_tuple=False)
        valid_images = images[instance_mask]
        if instance_chunk_size is None:
            features = self.encoder(valid_images)
        else:
            if instance_chunk_size < 1:
                raise ValueError("instance_chunk_size must be positive")
            features = torch.cat(
                [
                    self.encoder(valid_images[start : start + instance_chunk_size])
                    for start in range(0, len(valid_images), instance_chunk_size)
                ]
            )

        patient_sums = features.new_zeros((images.shape[0], features.shape[1]))
        patient_sums.index_add_(0, valid_positions[:, 0], features)
        counts = instance_mask.sum(dim=1, keepdim=True).to(features.dtype)
        patient_features = patient_sums / counts
        patient_logits = self.head(self.dropout(patient_features))
        return {
            "patient_logits": patient_logits,
            "patient_probabilities": torch.softmax(patient_logits, dim=1),
            "patient_log_probabilities": torch.log_softmax(patient_logits, dim=1),
            "valid_positions": valid_positions,
        }
