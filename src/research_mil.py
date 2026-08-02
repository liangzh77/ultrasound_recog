"""Gated attention pooling for patient-level multiple-instance learning."""

from __future__ import annotations

import math

import torch
from torch import nn


class GatedAttentionMILClassifier(nn.Module):
    """Aggregate a variable-length image bag into one patient prediction.

    The attention weights are normalized within each patient bag. They are
    model importance scores, not lesion localizations or causal explanations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        attention_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive")
        self.encoder = encoder
        self.attention_tanh = nn.Linear(feature_dim, attention_dim)
        self.attention_sigmoid = nn.Linear(feature_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def _encode(
        self,
        images: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_chunk_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return features, valid_positions

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

        valid_features, valid_positions = self._encode(
            images,
            instance_mask,
            instance_chunk_size,
        )
        batch_size, max_instances = instance_mask.shape
        feature_dim = valid_features.shape[1]
        padded_features = valid_features.new_zeros(
            (batch_size, max_instances, feature_dim)
        )
        padded_features[instance_mask] = valid_features

        gated = torch.tanh(self.attention_tanh(padded_features)) * torch.sigmoid(
            self.attention_sigmoid(padded_features)
        )
        attention_logits = self.attention_score(gated).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~instance_mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.masked_fill(~instance_mask, 0.0)

        patient_features = torch.sum(
            attention_weights.unsqueeze(-1) * padded_features,
            dim=1,
        )
        patient_logits = self.head(self.dropout(patient_features))
        patient_probabilities = torch.softmax(patient_logits, dim=1)
        return {
            "patient_logits": patient_logits,
            "patient_probabilities": patient_probabilities,
            "patient_log_probabilities": torch.log_softmax(patient_logits, dim=1),
            "attention_weights": attention_weights,
            "valid_positions": valid_positions,
        }


def summarize_attention(
    summaries: list[dict[str, object]],
    collapse_threshold: float = 0.95,
) -> dict[str, float | int]:
    """Summarize attention concentration without treating it as explanation."""
    if not 0.0 < collapse_threshold <= 1.0:
        raise ValueError("collapse_threshold must be in (0, 1]")
    if not summaries:
        raise ValueError("Attention summaries cannot be empty")

    maxima = []
    multi_image_maxima = []
    for item in summaries:
        weights = [float(value) for value in item["attention_weights"]]
        if not weights:
            raise ValueError("Every attention summary needs at least one weight")
        if abs(sum(weights) - 1.0) > 2e-3:
            raise ValueError("Attention weights must sum to one per patient")
        maximum = max(weights)
        maxima.append(maximum)
        if len(weights) > 1:
            multi_image_maxima.append(maximum)

    sorted_multi = sorted(multi_image_maxima)
    if sorted_multi:
        p95_index = math.ceil(0.95 * len(sorted_multi)) - 1
        mean_multi = sum(sorted_multi) / len(sorted_multi)
        collapse_rate = sum(
            value >= collapse_threshold for value in sorted_multi
        ) / len(sorted_multi)
        p95_multi = sorted_multi[p95_index]
    else:
        mean_multi = 1.0
        collapse_rate = 0.0
        p95_multi = 1.0
    return {
        "patients": len(summaries),
        "single_image_patients": len(summaries) - len(multi_image_maxima),
        "multi_image_patients": len(multi_image_maxima),
        "mean_max_attention_all": sum(maxima) / len(maxima),
        "mean_max_attention_multi_image": mean_multi,
        "p95_max_attention_multi_image": p95_multi,
        "collapse_threshold": collapse_threshold,
        "multi_image_collapse_rate": collapse_rate,
    }
